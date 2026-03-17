"""Simplification Quality Evaluator.

Measures:
- Before/after readability comparison
- Concept preservation (critical concepts from input → output)
- Information density changes
- Overall simplification quality score
"""

import logging
import re
from typing import Any

from models.evaluation import (
    ReadabilityComparison,
    SimplificationQualityMetrics,
)
from evaluators.base import BaseEvaluator

logger = logging.getLogger("evaluators.simplification")


# Critical patterns that MUST be preserved
DIAGNOSIS_PATTERNS = [
    r'\b(?:diagnosed with|diagnosis of|positive for|confirmed|impression:?)\s*[:\s]*([^\.]+)',
    r'\b(?:findings?:?)\s*[:\s]*([^\.]+)',
]

MEDICATION_PATTERNS = [
    r'\b(\d+\.?\d*)\s*(mg|mcg|g|ml|units?|tablets?|capsules?)\b',
    r'\b(aspirin|ibuprofen|acetaminophen|metformin|lisinopril|atorvastatin|omeprazole|amlodipine|metoprolol|losartan)\b',
]

MEASUREMENT_PATTERNS = [
    r'\b(\d+\.?\d*)\s*(mm|cm|m|kg|g|lbs?|pounds?|inches?|feet)\b',
    r'\b(\d+\.?\d*)\s*(mg/dL|g/dL|mmol/L|mEq/L|U/L|IU/L|ng/mL|pg/mL)\b',
    r'\b(\d+)\s*/\s*(\d+)\s*(?:mmHg)?\b',  # Blood pressure
    r'\b(\d+\.?\d*)\s*%\b',  # Percentages
    r'\b[CTLS]\d+(?:-[CTLS]?\d+)?\b',  # Spinal levels
    r'\b(\d+)\s*(?:bpm|beats per minute)\b',  # Heart rate
]


def _count_syllables(word: str) -> int:
    """Estimate syllable count for a word."""
    word = word.lower()
    if len(word) <= 3:
        return 1
    
    vowels = "aeiouy"
    count = 0
    prev_is_vowel = False
    
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_is_vowel:
            count += 1
        prev_is_vowel = is_vowel
    
    # Adjust for silent e
    if word.endswith('e') and count > 1:
        count -= 1
    
    return max(1, count)


def _compute_readability(text: str) -> dict:
    """Compute readability metrics for text."""
    # Tokenize
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    
    if not sentences or not words:
        return {
            "fkgl": 0.0,
            "smog": 0.0,
            "avg_sentence_length": 0.0,
            "word_count": 0,
            "sentence_count": 0,
        }
    
    word_count = len(words)
    sentence_count = len(sentences)
    avg_sentence_length = word_count / sentence_count
    
    # Count syllables
    total_syllables = sum(_count_syllables(w) for w in words)
    avg_syllables_per_word = total_syllables / word_count if word_count else 0
    
    # Count complex words (3+ syllables)
    complex_words = sum(1 for w in words if _count_syllables(w) >= 3)
    
    # FKGL formula
    fkgl = (0.39 * avg_sentence_length) + (11.8 * avg_syllables_per_word) - 15.59
    
    # SMOG formula (simplified)
    if sentence_count >= 3:
        smog = 1.0430 * ((complex_words * (30 / sentence_count)) ** 0.5) + 3.1291
    else:
        smog = 3.0 + (complex_words ** 0.5)
    
    return {
        "fkgl": round(max(0, fkgl), 1),
        "smog": round(max(0, smog), 1),
        "avg_sentence_length": round(avg_sentence_length, 1),
        "word_count": word_count,
        "sentence_count": sentence_count,
    }


def _extract_matches(text: str, patterns: list[str]) -> set[str]:
    """Extract all matches for given patterns."""
    matches = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            # Normalize: lowercase, strip whitespace
            normalized = match.group(0).strip().lower()
            normalized = " ".join(normalized.split())
            matches.add(normalized)
    return matches


def _compute_jargon_density(text: str, nlp=None, umls_index=None) -> float:
    """Compute medical jargon density."""
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    if not words:
        return 0.0
    
    # Simple heuristic: words > 8 chars that aren't common English
    common_words = {
        "the", "and", "that", "have", "for", "not", "with", "you", "this",
        "but", "his", "from", "they", "she", "which", "there", "been", "have",
        "their", "said", "each", "about", "would", "your", "when", "will",
        "patient", "treatment", "medical", "condition", "hospital", "doctor",
    }
    
    jargon_count = 0
    for word in words:
        if len(word) > 8 and word.lower() not in common_words:
            jargon_count += 1
    
    return jargon_count / len(words)


class SimplificationEvaluator(BaseEvaluator):
    """
    Evaluates simplification quality by comparing original vs simplified text.
    
    Computes:
    - Before/after readability metrics
    - Concept preservation rates
    - Critical information retention
    - Overall simplification quality score
    """
    
    name = "simplification_evaluator"
    category = "simplification"
    
    def __init__(self, opensearch_client=None):
        self._opensearch_client = opensearch_client
        self._umls_index = None
        self._nlp = None
    
    def _get_umls_index(self):
        """Lazy load UMLS index."""
        if self._umls_index is not None:
            return self._umls_index
        
        try:
            from services.umls_service import UMLSSynonymIndex
            from clients.opensearch import get_opensearch_client
            
            client = self._opensearch_client or get_opensearch_client()
            if client.connect():
                self._umls_index = UMLSSynonymIndex(client)
        except Exception as e:
            logger.debug(f"UMLS not available: {e}")
        
        return self._umls_index
    
    def _get_nlp(self):
        """Lazy load spaCy."""
        if self._nlp is None:
            try:
                import spacy
                self._nlp = spacy.load("en_core_web_sm")
            except Exception:
                pass
        return self._nlp
    
    def _extract_concepts(self, text: str) -> set[str]:
        """Extract medical concepts (CUIs) from text."""
        umls_index = self._get_umls_index()
        nlp = self._get_nlp()
        
        if not umls_index or not nlp:
            return set()
        
        doc = nlp(text)
        concepts = set()
        
        for chunk in doc.noun_chunks:
            matches = umls_index.search_term(chunk.text, limit=1)
            if matches and matches[0].score > 5.0:
                concepts.add(matches[0].cui)
        
        return concepts
    
    def evaluate_readability_comparison(
        self,
        original_text: str,
        simplified_text: str,
    ) -> ReadabilityComparison:
        """Compare readability before and after simplification."""
        original_metrics = _compute_readability(original_text)
        simplified_metrics = _compute_readability(simplified_text)
        
        original_jargon = _compute_jargon_density(original_text)
        simplified_jargon = _compute_jargon_density(simplified_text)
        
        return ReadabilityComparison(
            # Original metrics
            original_fkgl=original_metrics["fkgl"],
            original_smog=original_metrics["smog"],
            original_avg_sentence_length=original_metrics["avg_sentence_length"],
            original_jargon_density=round(original_jargon, 4),
            # Simplified metrics
            simplified_fkgl=simplified_metrics["fkgl"],
            simplified_smog=simplified_metrics["smog"],
            simplified_avg_sentence_length=simplified_metrics["avg_sentence_length"],
            simplified_jargon_density=round(simplified_jargon, 4),
            # Deltas
            fkgl_delta=round(simplified_metrics["fkgl"] - original_metrics["fkgl"], 1),
            smog_delta=round(simplified_metrics["smog"] - original_metrics["smog"], 1),
            sentence_length_delta=round(
                simplified_metrics["avg_sentence_length"] - original_metrics["avg_sentence_length"], 1
            ),
            jargon_reduction=round(original_jargon - simplified_jargon, 4),
        )
    
    def evaluate_simplification_quality(
        self,
        original_text: str,
        simplified_text: str,
    ) -> SimplificationQualityMetrics:
        """Evaluate overall simplification quality."""
        # Extract critical items from original
        original_diagnoses = _extract_matches(original_text, DIAGNOSIS_PATTERNS)
        original_medications = _extract_matches(original_text, MEDICATION_PATTERNS)
        original_measurements = _extract_matches(original_text, MEASUREMENT_PATTERNS)
        
        # Extract from simplified
        simplified_diagnoses = _extract_matches(simplified_text, DIAGNOSIS_PATTERNS)
        simplified_medications = _extract_matches(simplified_text, MEDICATION_PATTERNS)
        simplified_measurements = _extract_matches(simplified_text, MEASUREMENT_PATTERNS)
        
        # Find preserved items (using fuzzy matching for measurements)
        def fuzzy_preserved(original_set: set, simplified_set: set) -> tuple[int, set]:
            """Check preservation with fuzzy matching for numbers."""
            preserved = 0
            missing = set()
            for orig in original_set:
                # Extract numbers from original
                orig_nums = set(re.findall(r'\d+\.?\d*', orig))
                found = False
                for simp in simplified_set:
                    simp_nums = set(re.findall(r'\d+\.?\d*', simp))
                    if orig_nums and orig_nums & simp_nums:
                        found = True
                        break
                    if orig.lower() == simp.lower():
                        found = True
                        break
                if found:
                    preserved += 1
                else:
                    missing.add(orig)
            return preserved, missing
        
        diag_preserved, missing_diag = fuzzy_preserved(original_diagnoses, simplified_diagnoses)
        med_preserved, missing_meds = fuzzy_preserved(original_medications, simplified_medications)
        meas_preserved, missing_meas = fuzzy_preserved(original_measurements, simplified_measurements)
        
        # Calculate rates
        diagnosis_rate = diag_preserved / len(original_diagnoses) if original_diagnoses else 1.0
        medication_rate = med_preserved / len(original_medications) if original_medications else 1.0
        measurement_rate = meas_preserved / len(original_measurements) if original_measurements else 1.0
        
        # Concept preservation using UMLS
        original_concepts = self._extract_concepts(original_text)
        simplified_concepts = self._extract_concepts(simplified_text)
        shared_concepts = original_concepts & simplified_concepts
        concept_recall = len(shared_concepts) / len(original_concepts) if original_concepts else 1.0
        
        # Critical concept recall (weighted average of diagnosis, med, measurement)
        total_critical = len(original_diagnoses) + len(original_medications) + len(original_measurements)
        if total_critical > 0:
            critical_recall = (diag_preserved + med_preserved + meas_preserved) / total_critical
        else:
            critical_recall = 1.0
        
        # Information density
        original_words = len(re.findall(r'\b[a-zA-Z]+\b', original_text))
        simplified_words = len(re.findall(r'\b[a-zA-Z]+\b', simplified_text))
        
        original_concept_count = len(original_concepts) or 1
        simplified_concept_count = len(simplified_concepts) or 1
        
        original_density = original_words / original_concept_count
        simplified_density = simplified_words / simplified_concept_count
        
        expansion_ratio = simplified_words / original_words if original_words else 1.0
        
        # Compute overall simplification score (0-1)
        # Weights: concept preservation is most important
        readability_bonus = 0.0
        if simplified_words > 0:
            # Reward if simplified is more readable (lower FKGL)
            orig_read = _compute_readability(original_text)
            simp_read = _compute_readability(simplified_text)
            if orig_read["fkgl"] > simp_read["fkgl"]:
                readability_bonus = min(0.2, (orig_read["fkgl"] - simp_read["fkgl"]) / 10)
        
        simplification_score = (
            0.40 * critical_recall +
            0.30 * concept_recall +
            0.10 * measurement_rate +
            0.20 * readability_bonus
        )
        
        return SimplificationQualityMetrics(
            concept_recall=round(concept_recall, 4),
            critical_concept_recall=round(critical_recall, 4),
            measurement_preservation=round(measurement_rate, 4),
            original_words_per_concept=round(original_density, 2),
            simplified_words_per_concept=round(simplified_density, 2),
            expansion_ratio=round(expansion_ratio, 2),
            missing_diagnoses=list(missing_diag)[:5],  # Limit to 5
            missing_medications=list(missing_meds)[:5],
            missing_measurements=list(missing_meas)[:5],
            simplification_score=round(simplification_score, 4),
        )
    
    def evaluate_sample(
        self,
        original_text: str,
        simplified_text: str,
        reference_text: str | None = None,
        **kwargs: Any,
    ) -> tuple[ReadabilityComparison, SimplificationQualityMetrics]:
        """
        Evaluate simplification quality for a single sample.
        
        Returns:
            Tuple of (ReadabilityComparison, SimplificationQualityMetrics)
        """
        readability_comp = self.evaluate_readability_comparison(original_text, simplified_text)
        quality = self.evaluate_simplification_quality(original_text, simplified_text)
        
        return readability_comp, quality
