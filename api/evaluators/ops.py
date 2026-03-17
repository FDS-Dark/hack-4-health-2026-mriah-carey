"""E7: Operations Metrics Evaluator.

Tracks:
- Latency p50/p95
- Cost per report (tokens)
- Retry/refinement rate
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from models.evaluation import OpsMetrics
from evaluators.base import BaseEvaluator

logger = logging.getLogger("evaluators.ops")


@dataclass
class PipelineRun:
    """Metrics from a single pipeline run."""
    latency_ms: float
    input_tokens: int
    output_tokens: int
    retry_count: int
    refinement_count: int
    first_pass_success: bool


@dataclass
class OpsCollector:
    """Collector for ops metrics across multiple runs."""
    runs: list[PipelineRun] = field(default_factory=list)
    
    def add_run(
        self,
        latency_ms: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        retry_count: int = 0,
        refinement_count: int = 0,
        first_pass_success: bool = True,
    ):
        """Add a pipeline run to the collector."""
        self.runs.append(PipelineRun(
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            retry_count=retry_count,
            refinement_count=refinement_count,
            first_pass_success=first_pass_success,
        ))
    
    def compute_metrics(self) -> OpsMetrics:
        """Compute aggregate ops metrics."""
        if not self.runs:
            return OpsMetrics(
                latency_p50_ms=0.0,
                latency_p95_ms=0.0,
                latency_mean_ms=0.0,
                cost_per_report_tokens=0.0,
                retry_rate=0.0,
                refinement_rate=0.0,
                first_pass_success_rate=0.0,
            )
        
        n = len(self.runs)
        
        # Latency stats
        latencies = sorted(r.latency_ms for r in self.runs)
        p50_idx = int(n * 0.5)
        p95_idx = int(n * 0.95)
        
        latency_p50 = latencies[p50_idx]
        latency_p95 = latencies[min(p95_idx, n - 1)]
        latency_mean = sum(latencies) / n
        
        # Token costs
        total_tokens = sum(r.input_tokens + r.output_tokens for r in self.runs)
        cost_per_report = total_tokens / n
        
        # Retry/refinement rates
        runs_with_retry = sum(1 for r in self.runs if r.retry_count > 0)
        runs_with_refinement = sum(1 for r in self.runs if r.refinement_count > 0)
        first_pass_successes = sum(1 for r in self.runs if r.first_pass_success)
        
        retry_rate = runs_with_retry / n
        refinement_rate = runs_with_refinement / n
        first_pass_success_rate = first_pass_successes / n
        
        return OpsMetrics(
            latency_p50_ms=round(latency_p50, 2),
            latency_p95_ms=round(latency_p95, 2),
            latency_mean_ms=round(latency_mean, 2),
            cost_per_report_tokens=round(cost_per_report, 2),
            retry_rate=round(retry_rate, 4),
            refinement_rate=round(refinement_rate, 4),
            first_pass_success_rate=round(first_pass_success_rate, 4),
        )


class OpsEvaluator(BaseEvaluator):
    """
    Evaluates operational metrics for the pipeline.
    
    Unlike other evaluators, this one collects metrics from actual
    pipeline runs rather than analyzing text content.
    """
    
    name = "ops_evaluator"
    category = "ops"
    
    def __init__(self):
        """Initialize ops evaluator with a collector."""
        self.collector = OpsCollector()
    
    def record_run(
        self,
        latency_ms: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        retry_count: int = 0,
        refinement_count: int = 0,
        first_pass_success: bool = True,
    ):
        """Record metrics from a pipeline run."""
        self.collector.add_run(
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            retry_count=retry_count,
            refinement_count=refinement_count,
            first_pass_success=first_pass_success,
        )
    
    def evaluate_sample(
        self,
        original_text: str,
        simplified_text: str,
        reference_text: str | None = None,
        **kwargs: Any,
    ) -> OpsMetrics:
        """
        Return current aggregate ops metrics.
        
        Note: This doesn't evaluate the texts directly. Instead, it
        returns metrics collected via record_run().
        """
        return self.collector.compute_metrics()
    
    def get_metrics(self) -> OpsMetrics:
        """Get current aggregate ops metrics."""
        return self.collector.compute_metrics()
    
    def reset(self):
        """Reset the collector."""
        self.collector = OpsCollector()


class PipelineTimer:
    """Context manager for timing pipeline runs."""
    
    def __init__(self, ops_evaluator: OpsEvaluator | None = None):
        self.ops_evaluator = ops_evaluator
        self.start_time = 0.0
        self.end_time = 0.0
        self.input_tokens = 0
        self.output_tokens = 0
        self.retry_count = 0
        self.refinement_count = 0
        self.first_pass_success = True
    
    def __enter__(self):
        self.start_time = time.perf_counter() * 1000  # ms
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter() * 1000  # ms
        
        if self.ops_evaluator:
            self.ops_evaluator.record_run(
                latency_ms=self.end_time - self.start_time,
                input_tokens=self.input_tokens,
                output_tokens=self.output_tokens,
                retry_count=self.retry_count,
                refinement_count=self.refinement_count,
                first_pass_success=self.first_pass_success,
            )
    
    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return self.end_time - self.start_time
