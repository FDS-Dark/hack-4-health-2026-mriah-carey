"""E4: Concept Overlap Evaluator (UMLS-based).

Computes CUI precision/recall/F1 as metrics, weighted by semantic type.
"""

import logging
from typing import Any

from models.evaluation import ConceptOverlapMetrics
from evaluators.base import BaseEvaluator

logger = logging.getLogger("evaluators.concept")


# High-importance semantic types for weighting
HIGH_IMPORTANCE_TYPES = {
    "T047",  # Disease or Syndrome
    "T048",  # Mental or Behavioral Dysfunction
    "T184",  # Sign or Symptom
    "T121",  # Pharmacologic Substance
    "T200",  # Clinical Drug
    "T060",  # Diagnostic Procedure
    "T061",  # Therapeutic or Preventive Procedure
    "T033",  # Finding
    "T034",  # Laboratory or Test Result
}

WEIGHT_HIGH = 2.0
WEIGHT_DEFAULT = 1.0


class ConceptOverlapEvaluator(BaseEvaluator):
    """
    Evaluates concept overlap between original and simplified text.
    
    Uses UMLS CUI extraction and computes:
    - CUI precision: (simplified ∩ original) / simplified
    - CUI recall: (simplified ∩ original) / original
    - CUI F1: harmonic mean
    - Weighted F1 by semantic type importance
    """
    
    name = "concept_overlap_evaluator"
    category = "concept"
    
    def __init__(self, opensearch_client=None):
        """
        Initialize concept overlap evaluator.
        
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
                logger.info("UMLS index loaded for concept evaluation")
            else:
                logger.warning("OpenSearch not available for UMLS lookups")
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
    
    def _extract_cuis(self, text: str) -> dict[str, list[str]]:
        """
        Extract CUIs from text.
        
        Returns:
            Dict mapping CUI to list of semantic types.
        """
        umls_index = self._get_umls_index()
        if not umls_index:
            return {}
        
        nlp = self._get_nlp()
        if not nlp:
            return {}
        
        doc = nlp(text)
        cuis: dict[str, list[str]] = {}
        
        # Extract from noun chunks
        for chunk in doc.noun_chunks:
            matches = umls_index.search_term(chunk.text, limit=1)
            if matches and matches[0].score > 5.0:
                match = matches[0]
                cuis[match.cui] = match.semantic_types
        
        # Also check named entities
        for ent in doc.ents:
            if ent.label_ in {"ORG", "PRODUCT", "GPE"}:
                continue  # Skip non-medical entities
            matches = umls_index.search_term(ent.text, limit=1)
            if matches and matches[0].score > 5.0:
                match = matches[0]
                cuis[match.cui] = match.semantic_types
        
        return cuis
    
    def _compute_weighted_f1(
        self,
        original_cuis: dict[str, list[str]],
        simplified_cuis: dict[str, list[str]],
        shared_cuis: set[str],
    ) -> float:
        """Compute F1 weighted by semantic type importance."""
        if not original_cuis or not simplified_cuis:
            return 0.0
        
        def get_weight(cui: str, cui_dict: dict[str, list[str]]) -> float:
            types = cui_dict.get(cui, [])
            if any(t in HIGH_IMPORTANCE_TYPES for t in types):
                return WEIGHT_HIGH
            return WEIGHT_DEFAULT
        
        # Weighted recall
        original_weight = sum(get_weight(cui, original_cuis) for cui in original_cuis)
        shared_original_weight = sum(
            get_weight(cui, original_cuis) for cui in shared_cuis
        )
        weighted_recall = shared_original_weight / original_weight if original_weight else 0.0
        
        # Weighted precision
        simplified_weight = sum(get_weight(cui, simplified_cuis) for cui in simplified_cuis)
        shared_simplified_weight = sum(
            get_weight(cui, simplified_cuis) for cui in shared_cuis
        )
        weighted_precision = shared_simplified_weight / simplified_weight if simplified_weight else 0.0
        
        # F1
        if weighted_precision + weighted_recall == 0:
            return 0.0
        
        return 2 * (weighted_precision * weighted_recall) / (weighted_precision + weighted_recall)
    
    def evaluate_sample(
        self,
        original_text: str,
        simplified_text: str,
        reference_text: str | None = None,
        **kwargs: Any,
    ) -> ConceptOverlapMetrics:
        """Evaluate concept overlap between original and simplified."""
        logger.debug("Extracting CUIs for concept overlap evaluation")
        
        original_cuis = self._extract_cuis(original_text)
        simplified_cuis = self._extract_cuis(simplified_text)
        
        original_set = set(original_cuis.keys())
        simplified_set = set(simplified_cuis.keys())
        shared = original_set & simplified_set
        
        # Compute metrics
        precision = len(shared) / len(simplified_set) if simplified_set else 0.0
        recall = len(shared) / len(original_set) if original_set else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0.0
        
        weighted_f1 = self._compute_weighted_f1(original_cuis, simplified_cuis, shared)
        
        return ConceptOverlapMetrics(
            cui_precision=round(precision, 4),
            cui_recall=round(recall, 4),
            cui_f1=round(f1, 4),
            original_cui_count=len(original_set),
            simplified_cui_count=len(simplified_set),
            shared_cui_count=len(shared),
            semantic_type_weighted_f1=round(weighted_f1, 4) if weighted_f1 else None,
        )
