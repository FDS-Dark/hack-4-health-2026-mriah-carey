"""V3: UMLS grounding precision validator."""

import logging
import re
from dataclasses import dataclass
from functools import lru_cache

from clients.opensearch import OpenSearchClient, get_opensearch_client
from models.validation import (
    ErrorCode,
    Severity,
    ValidationError,
    ValidationResult,
)
from validators.base import BaseValidator

logger = logging.getLogger("validators.umls_grounding")


@dataclass
class ConceptMatch:
    """A matched UMLS concept."""
    cui: str
    term: str
    semantic_types: list[str]
    score: float


# High-risk semantic types - ONLY truly dangerous medical categories
# These are semantic types where introducing a new concept could cause harm
HIGH_RISK_SEMANTIC_TYPES = {
    "T191",  # Neoplastic Process (cancers)
}

# High-risk terms - ONLY specific dangerous diagnoses
# These are terms that if introduced incorrectly could cause patient harm
HIGH_RISK_TERMS = {
    # Cancers - definitive diagnosis terms
    "cancer", "carcinoma", "malignant", "malignancy", "neoplasm", 
    "metastasis", "metastatic", "metastases", "lymphoma", "leukemia",
    "sarcoma", "melanoma", "adenocarcinoma",
    
    # Cardiovascular emergencies
    "stroke", "cerebrovascular accident", "cva",
    "myocardial infarction", "heart attack", "mi",
    "pulmonary embolism", "pulmonary embolus",
    "deep vein thrombosis", "dvt",
    "aortic dissection", "ruptured aneurysm",
    
    # Other life-threatening conditions
    "sepsis", "septic shock",
    "respiratory failure", "cardiac arrest",
    "brain death", "terminal",
}

# Additional medical context words not in spaCy stop words but safe in simplified text
# These supplement spaCy's stop words for medical document context
MEDICAL_CONTEXT_ALLOWLIST = {
    # Common medical procedure words (not diagnoses)
    "procedure", "surgery", "operation", "treatment", "therapy",
    "reconstruction", "repair", "technique", "method", "incision",
    
    # Anatomical terms commonly used in simplification (not diagnoses)
    "chest", "body", "skin", "tissue", "bone", "sternum", "ribs",
    "breastbone", "nerves", "wall",
    
    # Common simplified text patterns
    "doctor", "surgeon", "hospital", "clinic", "patient",
    "recovery", "discomfort", "pain",
    
    # Procedure types (not diagnoses)
    "thoracoscopic", "minimally", "invasive", "cryotherapy",
    
    # Anatomical conditions being treated (context-dependent, not new diagnoses)
    "pectus", "excavatum", "carinatum", "deformity", "deformities",
}


@lru_cache(maxsize=1)
def _get_stop_words() -> frozenset[str]:
    """Get combined stop words from spaCy and medical context allowlist."""
    stop_words: set[str] = set()
    
    try:
        from spacy.lang.en.stop_words import STOP_WORDS
        stop_words.update(STOP_WORDS)
        logger.debug(f"Loaded {len(STOP_WORDS)} spaCy stop words")
    except ImportError:
        logger.warning("Could not import spaCy stop words, using minimal fallback")
        # Minimal fallback if spaCy not available
        stop_words.update({
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "can", "must", "shall", "this", "that",
            "these", "those", "i", "you", "he", "she", "it", "we", "they",
        })
    
    # Add medical context allowlist
    stop_words.update(MEDICAL_CONTEXT_ALLOWLIST)
    
    return frozenset(stop_words)


class UMLSGroundingValidator(BaseValidator):
    """
    Validates UMLS concept grounding precision.
    
    This validator focuses on catching DANGEROUS new terms, not all new terms.
    A simplified report naturally uses different words than the original -
    we only care about introducing serious diagnoses that weren't there.
    
    Checks:
    - Summary doesn't introduce high-risk diagnoses (cancer, stroke, etc.) not in original
    - Uses spaCy stop words + medical context allowlist to avoid false positives
    """
    
    name = "umls_grounding_validator"
    
    def __init__(
        self,
        opensearch_client: OpenSearchClient | None = None,
        precision_threshold: float = 0.3,  # Much lower - simplification naturally changes words
        min_term_length: int = 4,  # Ignore very short terms
    ):
        self._client = opensearch_client
        self.precision_threshold = precision_threshold
        self.min_term_length = min_term_length
        self._stop_words = _get_stop_words()
    
    @property
    def client(self) -> OpenSearchClient:
        if self._client is None:
            self._client = get_opensearch_client()
        return self._client
    
    def _is_allowed_term(self, term: str) -> bool:
        """Check if a term is on the allowlist (should not be flagged)."""
        term_lower = term.lower().strip()
        
        # Check exact match against stop words
        if term_lower in self._stop_words:
            return True
        
        # Check if it's too short
        if len(term_lower) < self.min_term_length:
            return True
        
        # Check if all words in the term are stop words or very short
        words = term_lower.split()
        if all(w in self._stop_words or len(w) < 3 for w in words):
            return True
        
        return False
    
    def _is_high_risk_term(self, term: str) -> bool:
        """Check if a term is genuinely high-risk (dangerous diagnosis)."""
        term_lower = term.lower().strip()
        
        # Only flag if it matches a specific high-risk term
        for risk_term in HIGH_RISK_TERMS:
            # Exact match or term contains the risk term as a whole word
            if risk_term == term_lower:
                return True
            if re.search(rf'\b{re.escape(risk_term)}\b', term_lower):
                return True
        
        return False
    
    def _extract_medical_terms(self, text: str) -> list[str]:
        """Extract potential medical terms from text - focused extraction."""
        # More conservative extraction - only multi-word phrases and longer words
        terms = set()
        
        # Extract longer words (5+ chars) that might be medical terms
        words = re.findall(r'\b[a-zA-Z]{5,}\b', text.lower())
        
        for word in words:
            if word not in self._stop_words and not self._is_allowed_term(word):
                terms.add(word)
        
        # Extract 2-3 word medical phrases
        text_lower = text.lower()
        # Only look for specific medical phrase patterns
        phrase_patterns = [
            r'\b([a-z]+\s+(?:cancer|carcinoma|tumor|syndrome|disease|disorder|failure))\b',
            r'\b((?:acute|chronic|severe|mild)\s+[a-z]+)\b',
            r'\b([a-z]+\s+(?:infarction|embolism|thrombosis|hemorrhage))\b',
        ]
        
        for pattern in phrase_patterns:
            for match in re.finditer(pattern, text_lower):
                phrase = match.group(1)
                if not self._is_allowed_term(phrase):
                    terms.add(phrase)
        
        return list(terms)
    
    def _lookup_cuis(self, terms: list[str]) -> dict[str, ConceptMatch]:
        """Look up CUIs for a list of terms."""
        cui_map = {}
        
        if not self.client.is_connected:
            if not self.client.connect():
                logger.warning("OpenSearch not connected - skipping UMLS lookup")
                return cui_map
        
        for term in terms:
            hits = self.client.search_term(term, limit=1)
            if hits:
                hit = hits[0]
                src = hit.get("_source", {})
                # Only include if it's a good match (high score)
                score = hit.get("_score", 0.0)
                if score >= 5.0:  # Require good match
                    cui_map[term] = ConceptMatch(
                        cui=src.get("cui", ""),
                        term=src.get("term", term),
                        semantic_types=src.get("semantic_types", []),
                        score=score
                    )
        
        return cui_map
    
    def _is_high_risk_semantic_type(self, semantic_types: list[str]) -> bool:
        """Check if any semantic type is high-risk."""
        return bool(set(semantic_types) & HIGH_RISK_SEMANTIC_TYPES)
    
    def validate(self, original_text: str, simplified_text: str) -> ValidationResult:
        """Validate that no dangerous diagnoses were introduced."""
        logger.info(f"{'='*60}")
        logger.info(f"UMLS GROUNDING VALIDATOR - Starting validation")
        logger.info(f"{'='*60}")
        
        errors: list[ValidationError] = []
        
        # Extract terms (focused extraction)
        orig_terms = self._extract_medical_terms(original_text)
        simp_terms = self._extract_medical_terms(simplified_text)
        
        logger.debug(f"Original medical terms ({len(orig_terms)}): {orig_terms[:10]}...")
        logger.debug(f"Simplified medical terms ({len(simp_terms)}): {simp_terms[:10]}...")
        
        # Check for high-risk terms in simplified that aren't in original
        # This is a keyword-based check, no UMLS needed
        orig_text_lower = original_text.lower()
        
        for term in simp_terms:
            if self._is_high_risk_term(term):
                # Check if this term appears in original
                if term.lower() not in orig_text_lower:
                    logger.error(f"HIGH-RISK DIAGNOSIS INTRODUCED: '{term}'")
                    errors.append(ValidationError(
                        code=ErrorCode.UMLS_HIGH_RISK_TERM,
                        severity=Severity.CRITICAL,
                        message=f"Dangerous diagnosis '{term}' introduced but not in original",
                        summary_span=term,
                        fix=f"Remove the term '{term}' - this is a serious diagnosis not mentioned in the original"
                    ))
        
        # Only do UMLS lookup if connected (optional enhancement)
        if self.client.is_connected or self.client.connect():
            logger.info("Checking UMLS concepts...")
            
            # Look up CUIs for high-risk terms only
            high_risk_in_simp = [t for t in simp_terms if self._is_high_risk_term(t)]
            
            if high_risk_in_simp:
                simp_cuis = self._lookup_cuis(high_risk_in_simp)
                orig_cuis = self._lookup_cuis([t for t in orig_terms if self._is_high_risk_term(t)])
                
                orig_cui_set = {m.cui for m in orig_cuis.values() if m.cui}
                
                for term, match in simp_cuis.items():
                    if match.cui and match.cui not in orig_cui_set:
                        if self._is_high_risk_semantic_type(match.semantic_types):
                            logger.error(f"HIGH-RISK CUI INTRODUCED: '{term}' (CUI: {match.cui})")
                            errors.append(ValidationError(
                                code=ErrorCode.UMLS_HIGH_RISK_TERM,
                                severity=Severity.CRITICAL,
                                message=f"High-risk concept '{term}' (CUI: {match.cui}) not in original",
                                summary_span=term,
                                metadata={"cui": match.cui, "semantic_types": match.semantic_types},
                                fix=f"Remove or verify the term '{term}'"
                            ))
            else:
                logger.debug("No high-risk terms found in simplified text")
        else:
            logger.debug("OpenSearch not available - using keyword-based validation only")
        
        if errors:
            logger.warning(f"UMLS GROUNDING VALIDATOR - FAILED with {len(errors)} errors")
            return ValidationResult.failure(self.name, errors)
        
        logger.info("UMLS GROUNDING VALIDATOR - PASSED ✓")
        return ValidationResult.success(self.name)
