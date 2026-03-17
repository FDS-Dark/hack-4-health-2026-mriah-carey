"""E5: NLI Distribution Evaluator.

Computes entailment/neutral/contradiction rates across sentence pairs.
Tracks rates to compare prompts/models even if contradictions are gated at runtime.
"""

import logging
from typing import Any

from models.evaluation import NLIDistributionMetrics
from evaluators.base import BaseEvaluator

logger = logging.getLogger("evaluators.nli")


class NLIDistributionEvaluator(BaseEvaluator):
    """
    Evaluates NLI distribution between original and simplified text.
    
    For each simplified sentence, finds the best matching original sentence
    and classifies the relationship as entailment/neutral/contradiction.
    
    This is useful for comparing prompts/models across versions.
    """
    
    name = "nli_distribution_evaluator"
    category = "nli"
    
    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        min_match_score: float = 0.25,
    ):
        """
        Initialize NLI distribution evaluator.
        
        Args:
            model_name: HuggingFace model for NLI.
            min_match_score: Minimum entity overlap to pair sentences.
        """
        self.model_name = model_name
        self.min_match_score = min_match_score
        self._model = None
        self._tokenizer = None
    
    def _load_model(self):
        """Lazy load NLI model."""
        if self._model is not None:
            return True
        
        logger.info(f"Loading NLI model: {self.model_name}")
        
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            logger.info("NLI model loaded successfully")
            return True
        except Exception as e:
            logger.warning(f"Failed to load NLI model: {e}")
            return False
    
    def _run_nli(self, premise: str, hypothesis: str) -> tuple[str, float]:
        """Run NLI on a single pair. Returns (label, confidence)."""
        if not self._load_model():
            return ("neutral", 0.5)
        
        import torch
        
        inputs = self._tokenizer(
            premise, hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        
        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]
        
        labels = ["contradiction", "neutral", "entailment"]
        max_idx = probs.argmax().item()
        
        return (labels[max_idx], probs[max_idx].item())
    
    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        try:
            from utils.nlp import split_into_sentences
            return split_into_sentences(text)
        except ImportError:
            import re
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if s.strip()]
    
    def _find_best_match(
        self,
        hypothesis: str,
        candidates: list[str],
    ) -> tuple[str | None, float]:
        """Find best matching sentence from candidates."""
        try:
            from utils.nlp import find_best_matching_sentence
            return find_best_matching_sentence(
                hypothesis, candidates, min_similarity=self.min_match_score
            )
        except ImportError:
            # Simple fallback: first sentence with word overlap
            hyp_words = set(hypothesis.lower().split())
            best_match = None
            best_score = 0.0
            
            for cand in candidates:
                cand_words = set(cand.lower().split())
                if not cand_words:
                    continue
                overlap = len(hyp_words & cand_words) / len(cand_words)
                if overlap > best_score:
                    best_score = overlap
                    best_match = cand
            
            if best_score >= self.min_match_score:
                return (best_match, best_score)
            return (None, 0.0)
    
    def evaluate_sample(
        self,
        original_text: str,
        simplified_text: str,
        reference_text: str | None = None,
        **kwargs: Any,
    ) -> NLIDistributionMetrics:
        """Evaluate NLI distribution for a single sample."""
        original_sentences = self._split_sentences(original_text)
        simplified_sentences = self._split_sentences(simplified_text)
        
        if not original_sentences or not simplified_sentences:
            return NLIDistributionMetrics(
                entailment_rate=0.0,
                neutral_rate=0.0,
                contradiction_rate=0.0,
                total_pairs=0,
                avg_entailment_confidence=0.0,
                avg_contradiction_confidence=0.0,
            )
        
        counts = {"entailment": 0, "neutral": 0, "contradiction": 0}
        confidences = {"entailment": [], "contradiction": []}
        total_pairs = 0
        
        for simp_sent in simplified_sentences:
            orig_sent, match_score = self._find_best_match(simp_sent, original_sentences)
            
            if orig_sent is None:
                continue
            
            label, confidence = self._run_nli(orig_sent, simp_sent)
            counts[label] += 1
            total_pairs += 1
            
            if label in confidences:
                confidences[label].append(confidence)
        
        if total_pairs == 0:
            return NLIDistributionMetrics(
                entailment_rate=0.0,
                neutral_rate=0.0,
                contradiction_rate=0.0,
                total_pairs=0,
                avg_entailment_confidence=0.0,
                avg_contradiction_confidence=0.0,
            )
        
        avg_ent_conf = (
            sum(confidences["entailment"]) / len(confidences["entailment"])
            if confidences["entailment"] else 0.0
        )
        avg_con_conf = (
            sum(confidences["contradiction"]) / len(confidences["contradiction"])
            if confidences["contradiction"] else 0.0
        )
        
        return NLIDistributionMetrics(
            entailment_rate=round(counts["entailment"] / total_pairs, 4),
            neutral_rate=round(counts["neutral"] / total_pairs, 4),
            contradiction_rate=round(counts["contradiction"] / total_pairs, 4),
            total_pairs=total_pairs,
            avg_entailment_confidence=round(avg_ent_conf, 4),
            avg_contradiction_confidence=round(avg_con_conf, 4),
        )
