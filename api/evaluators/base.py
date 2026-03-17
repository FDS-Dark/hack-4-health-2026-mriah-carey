"""Base evaluator interface for all evaluation metrics."""

import logging
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger("evaluators")


class BaseEvaluator(ABC):
    """Abstract base class for all evaluators."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this evaluator."""
        ...
    
    @property
    @abstractmethod
    def category(self) -> str:
        """Category of this evaluator (e.g., readability, semantic)."""
        ...
    
    @abstractmethod
    def evaluate_sample(
        self,
        original_text: str,
        simplified_text: str,
        reference_text: str | None = None,
        **kwargs: Any,
    ) -> BaseModel:
        """
        Evaluate a single sample.
        
        Args:
            original_text: The original medical report text.
            simplified_text: The simplified version to evaluate.
            reference_text: Optional reference simplification for comparison.
            **kwargs: Additional evaluator-specific arguments.
        
        Returns:
            Pydantic model with evaluation metrics.
        """
        ...
    
    def evaluate_batch(
        self,
        samples: list[dict],
        **kwargs: Any,
    ) -> list[BaseModel]:
        """
        Evaluate a batch of samples.
        
        Default implementation calls evaluate_sample for each.
        Override for batch-optimized implementations.
        
        Args:
            samples: List of dicts with 'original_text', 'simplified_text', 
                     and optionally 'reference_text'.
            **kwargs: Additional evaluator-specific arguments.
        
        Returns:
            List of evaluation results.
        """
        results = []
        for sample in samples:
            result = self.evaluate_sample(
                original_text=sample["original_text"],
                simplified_text=sample["simplified_text"],
                reference_text=sample.get("reference_text"),
                **kwargs,
            )
            results.append(result)
        return results
