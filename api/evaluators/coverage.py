"""E6: Coverage Evaluator.

For reference-based evaluation:
- Reference CUI recall
- Key measurements preservation
"""

import logging
import re
from typing import Any

from models.evaluation import CoverageMetrics
from evaluators.base import BaseEvaluator

logger = logging.getLogger("evaluators.coverage")


class CoverageEvaluator(BaseEvaluator):
    """
    Evaluates coverage of reference concepts and measurements.
    
    Useful when you have 1:1 target references to compare against.
    """
    
    name = "coverage_evaluator"
    category = "coverage"
    
    def __init__(self, opensearch_client=None):
        """
        Initialize coverage evaluator.
        
        Args:
            opensearch_client: OpenSearch client for UMLS lookups.
        """
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
            logger.warning(f"Failed to load UMLS index: {e}")
        
        return self._umls_index
    
    def _get_nlp(self):
        """Lazy load spaCy."""
        if self._nlp is None:
            try:
                import spacy
                self._nlp = spacy.load("en_core_web_sm")
            except Exception as e:
                logger.warning(f"Failed to load spaCy: {e}")
        return self._nlp
    
    def _extract_cuis(self, text: str) -> set[str]:
        """Extract CUIs from text."""
        umls_index = self._get_umls_index()
        if not umls_index:
            return set()
        
        nlp = self._get_nlp()
        if not nlp:
            return set()
        
        doc = nlp(text)
        cuis = set()
        
        for chunk in doc.noun_chunks:
            matches = umls_index.search_term(chunk.text, limit=1)
            if matches and matches[0].score > 5.0:
                cuis.add(matches[0].cui)
        
        return cuis
    
    def _extract_measurements(self, text: str) -> set[str]:
        """
        Extract key measurements from text.
        
        Returns normalized measurement strings for comparison.
        """
        patterns = [
            # Numeric with units
            r'(\d+\.?\d*)\s*(mm|cm|m|kg|g|mg|μg|ml|L|dL|mL|%|mmHg|bpm)',
            # Ranges
            r'(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\s*(mm|cm|m|kg|g|mg|μg|ml|L|dL|mL|%)',
            # Fractions
            r'(\d+)\s*/\s*(\d+)',
            # Lab values
            r'(\d+\.?\d*)\s*(U/L|mEq/L|mmol/L|mg/dL|g/dL)',
            # Spinal levels
            r'\b(C|T|L|S)\d+-?\d*\b',
            # Times/durations
            r'(\d+)\s*(hours?|days?|weeks?|months?|years?)',
        ]
        
        measurements = set()
        text_lower = text.lower()
        
        for pattern in patterns:
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                # Normalize the measurement
                normalized = re.sub(r'\s+', '', match.group(0).lower())
                measurements.add(normalized)
        
        return measurements
    
    def evaluate_sample(
        self,
        original_text: str,
        simplified_text: str,
        reference_text: str | None = None,
        **kwargs: Any,
    ) -> CoverageMetrics:
        """
        Evaluate coverage of ORIGINAL concepts and measurements in simplified output.
        
        This measures CONCEPT PRESERVATION - how much of the original medical 
        text's information is preserved in the simplified output.
        
        For pipeline evaluation, we care about:
        - original_text: The medical report that was input to the pipeline
        - simplified_text: The pipeline's output
        - We compare simplified vs ORIGINAL to measure information preservation
        """
        # Extract CUIs from ORIGINAL - we want to measure how much of the 
        # original medical information is preserved in the simplified output
        original_cuis = self._extract_cuis(original_text)
        simplified_cuis = self._extract_cuis(simplified_text)
        covered_cuis = original_cuis & simplified_cuis
        
        # This is ORIGINAL CUI recall - what fraction of original concepts survived
        cui_recall = (
            len(covered_cuis) / len(original_cuis) 
            if original_cuis else 1.0
        )
        
        logger.debug(
            f"CUI coverage: {len(covered_cuis)}/{len(original_cuis)} "
            f"({cui_recall:.1%}) original concepts preserved"
        )
        
        # Extract measurements from ORIGINAL 
        # (measurements like "5mm", "120/80" must be preserved)
        original_measurements = self._extract_measurements(original_text)
        simplified_measurements = self._extract_measurements(simplified_text)
        preserved = original_measurements & simplified_measurements
        
        measurement_rate = (
            len(preserved) / len(original_measurements)
            if original_measurements else 1.0
        )
        
        logger.debug(
            f"Measurement coverage: {len(preserved)}/{len(original_measurements)} "
            f"({measurement_rate:.1%}) measurements preserved"
        )
        
        return CoverageMetrics(
            reference_cui_recall=round(cui_recall, 4),  # Actually original CUI recall
            key_measurements_present=round(measurement_rate, 4),
            total_reference_cuis=len(original_cuis),
            covered_reference_cuis=len(covered_cuis),
            total_measurements=len(original_measurements),
            preserved_measurements=len(preserved),
        )
