"""E1: Readability & Accessibility Evaluator.

Computes:
- FKGL (Flesch-Kincaid Grade Level)
- SMOG Index
- Average sentence length
- Jargon density (CUIs / tokens)
"""

import logging
import re
import math

from models.evaluation import ReadabilityMetrics
from evaluators.base import BaseEvaluator

logger = logging.getLogger("evaluators.readability")


def count_syllables(word: str) -> int:
    """Count syllables in a word using a simple heuristic."""
    word = word.lower().strip()
    if not word:
        return 0
    
    # Handle common exceptions
    if len(word) <= 3:
        return 1
    
    # Count vowel groups
    vowels = "aeiouy"
    count = 0
    prev_was_vowel = False
    
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_was_vowel:
            count += 1
        prev_was_vowel = is_vowel
    
    # Adjust for silent 'e'
    if word.endswith("e") and count > 1:
        count -= 1
    
    # Adjust for common suffixes
    if word.endswith("le") and len(word) > 2 and word[-3] not in vowels:
        count += 1
    
    return max(1, count)


def count_complex_words(words: list[str]) -> int:
    """Count words with 3+ syllables (for SMOG)."""
    return sum(1 for w in words if count_syllables(w) >= 3)


def tokenize_words(text: str) -> list[str]:
    """Extract words from text."""
    words = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", text)
    return [w for w in words if len(w) > 0]


def split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    # Simple sentence splitting on common terminators
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]


class ReadabilityEvaluator(BaseEvaluator):
    """
    Evaluates readability and accessibility of simplified text.
    
    Metrics:
    - FKGL: Flesch-Kincaid Grade Level (lower = more readable)
    - SMOG: Simple Measure of Gobbledygook (estimates years of education needed)
    - Jargon density: Medical term density using UMLS CUI detection
    """
    
    name = "readability_evaluator"
    category = "readability"
    
    def __init__(self, umls_client=None):
        """
        Initialize the readability evaluator.
        
        Args:
            umls_client: Optional OpenSearch client for UMLS lookups.
        """
        self._umls_client = umls_client
        self._nlp = None
    
    def _get_nlp(self):
        """Lazy load spaCy NLP model."""
        if self._nlp is None:
            try:
                import spacy
                self._nlp = spacy.load("en_core_web_sm")
            except Exception as e:
                logger.warning(f"Failed to load spaCy: {e}")
        return self._nlp
    
    def _compute_fkgl(self, words: list[str], sentences: list[str]) -> float:
        """Compute Flesch-Kincaid Grade Level."""
        if not words or not sentences:
            return 0.0
        
        total_words = len(words)
        total_sentences = len(sentences)
        total_syllables = sum(count_syllables(w) for w in words)
        
        # FKGL formula
        fkgl = (
            0.39 * (total_words / total_sentences)
            + 11.8 * (total_syllables / total_words)
            - 15.59
        )
        return max(0, fkgl)
    
    def _compute_smog(self, words: list[str], sentences: list[str]) -> float:
        """Compute SMOG Index."""
        if not sentences:
            return 0.0
        
        complex_words = count_complex_words(words)
        sentence_count = len(sentences)
        
        # SMOG formula (requires 30+ sentences for accuracy, but we'll compute anyway)
        if sentence_count < 30:
            # Adjust for short texts
            complex_adjusted = complex_words * (30 / sentence_count)
        else:
            complex_adjusted = complex_words
        
        smog = 1.0430 * math.sqrt(complex_adjusted) + 3.1291
        return max(0, smog)
    
    def _count_cuis(self, text: str) -> int:
        """Count UMLS CUIs in text using NER."""
        if not self._umls_client:
            # Fallback: count medical-looking terms
            return self._count_medical_terms_heuristic(text)
        
        try:
            from services.umls_service import UMLSSynonymIndex
            umls_index = UMLSSynonymIndex(self._umls_client)
            
            nlp = self._get_nlp()
            if not nlp:
                return self._count_medical_terms_heuristic(text)
            
            doc = nlp(text)
            cui_count = 0
            
            # Check noun chunks for UMLS matches
            for chunk in doc.noun_chunks:
                matches = umls_index.search_term(chunk.text, limit=1)
                if matches and matches[0].score > 5.0:
                    cui_count += 1
            
            return cui_count
        except Exception as e:
            logger.warning(f"UMLS lookup failed: {e}")
            return self._count_medical_terms_heuristic(text)
    
    def _count_medical_terms_heuristic(self, text: str) -> int:
        """Heuristic count of medical terms without UMLS."""
        # Common medical suffixes/patterns
        medical_patterns = [
            r'\b\w+itis\b',      # inflammation
            r'\b\w+osis\b',      # condition
            r'\b\w+ectomy\b',    # surgical removal
            r'\b\w+opathy\b',    # disease
            r'\b\w+emia\b',      # blood condition
            r'\b\w+oma\b',       # tumor
            r'\b\w+scopy\b',     # examination
            r'\b\w+graphy\b',    # imaging
            r'\b\d+\s*mm\b',     # measurements
            r'\b\d+\s*cm\b',
            r'\b\d+\s*mg\b',
            r'\bC\d+-\d+\b',     # spinal levels
            r'\bT\d+-\d+\b',
            r'\bL\d+-\d+\b',
        ]
        
        count = 0
        text_lower = text.lower()
        for pattern in medical_patterns:
            count += len(re.findall(pattern, text_lower, re.IGNORECASE))
        
        return count
    
    def evaluate_sample(
        self,
        original_text: str,
        simplified_text: str,
        reference_text: str | None = None,
        **kwargs,
    ) -> ReadabilityMetrics:
        """Evaluate readability of simplified text."""
        logger.debug(f"Evaluating readability for {len(simplified_text)} chars")
        
        words = tokenize_words(simplified_text)
        sentences = split_sentences(simplified_text)
        
        total_tokens = len(words)
        avg_sentence_length = len(words) / len(sentences) if sentences else 0.0
        
        fkgl = self._compute_fkgl(words, sentences)
        smog = self._compute_smog(words, sentences)
        
        cui_count = self._count_cuis(simplified_text)
        jargon_density = cui_count / total_tokens if total_tokens > 0 else 0.0
        
        return ReadabilityMetrics(
            fkgl=round(fkgl, 2),
            smog=round(smog, 2),
            avg_sentence_length=round(avg_sentence_length, 2),
            jargon_density=round(jargon_density, 4),
            total_tokens=total_tokens,
            cui_count=cui_count,
        )
