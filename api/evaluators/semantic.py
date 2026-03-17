"""E3: Semantic Similarity Evaluator using BERTScore.

Uses biomedical encoders (PubMedBERT or BioClinicalBERT) for domain-specific
semantic similarity computation.

This is "meaning-ish similarity," not safety.
"""

import logging
from typing import Any

from models.evaluation import SemanticSimilarityMetrics
from evaluators.base import BaseEvaluator

logger = logging.getLogger("evaluators.semantic")


# Recommended biomedical encoders
BIOMEDICAL_ENCODERS = {
    "pubmedbert": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
    "bioclinicalbert": "emilyalsentzer/Bio_ClinicalBERT",
    "default": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
}


class SemanticSimilarityEvaluator(BaseEvaluator):
    """
    Evaluates semantic similarity using BERTScore with biomedical encoders.
    
    Recommended models:
    - PubMedBERT: Good for general biomedical text
    - BioClinicalBERT: Better for clinical notes
    """
    
    name = "semantic_similarity_evaluator"
    category = "semantic"
    
    def __init__(
        self,
        model_name: str = "pubmedbert",
        device: str | None = None,
        rescale_with_baseline: bool = True,
    ):
        """
        Initialize semantic similarity evaluator.
        
        Args:
            model_name: One of 'pubmedbert', 'bioclinicalbert', or full HF model name.
            device: Device to run model on. Auto-detected if None.
            rescale_with_baseline: Whether to rescale scores using baseline.
        """
        self.model_name = BIOMEDICAL_ENCODERS.get(model_name.lower(), model_name)
        self.device = device
        self.rescale_with_baseline = rescale_with_baseline
        self._scorer = None
    
    def _load_scorer(self):
        """Lazy load BERTScore scorer."""
        if self._scorer is not None:
            return True
        
        logger.info(f"Loading BERTScore with model: {self.model_name}")
        
        try:
            import bert_score
            # BERTScore uses lazy loading internally
            self._scorer = bert_score
            logger.info("BERTScore loaded successfully")
            return True
        except ImportError:
            logger.warning("bert-score not installed. Run: pip install bert-score")
            return False
        except Exception as e:
            logger.warning(f"Failed to load BERTScore: {e}")
            return False
    
    def evaluate_sample(
        self,
        original_text: str,
        simplified_text: str,
        reference_text: str | None = None,
        **kwargs: Any,
    ) -> SemanticSimilarityMetrics:
        """
        Evaluate semantic similarity between original and simplified text.
        
        If reference_text is provided, computes similarity to reference instead.
        """
        if not self._load_scorer():
            return SemanticSimilarityMetrics(
                precision=0.0, recall=0.0, f1=0.0, model_name=self.model_name
            )
        
        # Compare simplified to original (or reference if provided)
        reference = reference_text if reference_text else original_text
        
        try:
            P, R, F1 = self._scorer.score(
                cands=[simplified_text],
                refs=[reference],
                model_type=self.model_name,
                device=self.device,
                rescale_with_baseline=self.rescale_with_baseline,
                verbose=False,
            )
            
            return SemanticSimilarityMetrics(
                precision=round(P[0].item(), 4),
                recall=round(R[0].item(), 4),
                f1=round(F1[0].item(), 4),
                model_name=self.model_name,
            )
        except Exception as e:
            logger.warning(f"BERTScore evaluation failed: {e}")
            return SemanticSimilarityMetrics(
                precision=0.0, recall=0.0, f1=0.0, model_name=self.model_name
            )
    
    def evaluate_batch(
        self,
        samples: list[dict],
        **kwargs: Any,
    ) -> list[SemanticSimilarityMetrics]:
        """Batch evaluation for efficiency."""
        if not self._load_scorer():
            return [
                SemanticSimilarityMetrics(
                    precision=0.0, recall=0.0, f1=0.0, model_name=self.model_name
                )
                for _ in samples
            ]
        
        cands = [s["simplified_text"] for s in samples]
        refs = [s.get("reference_text") or s["original_text"] for s in samples]
        
        try:
            P, R, F1 = self._scorer.score(
                cands=cands,
                refs=refs,
                model_type=self.model_name,
                device=self.device,
                rescale_with_baseline=self.rescale_with_baseline,
                verbose=False,
            )
            
            return [
                SemanticSimilarityMetrics(
                    precision=round(p.item(), 4),
                    recall=round(r.item(), 4),
                    f1=round(f.item(), 4),
                    model_name=self.model_name,
                )
                for p, r, f in zip(P, R, F1)
            ]
        except Exception as e:
            logger.warning(f"BERTScore batch evaluation failed: {e}")
            return [
                SemanticSimilarityMetrics(
                    precision=0.0, recall=0.0, f1=0.0, model_name=self.model_name
                )
                for _ in samples
            ]
