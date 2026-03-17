"""MLflow service for logging evaluation metrics."""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from models.evaluation import (
    DatasetEvaluation,
    SampleEvaluation,
    ReadabilityMetrics,
    SemanticSimilarityMetrics,
    ConceptOverlapMetrics,
    NLIDistributionMetrics,
    CoverageMetrics,
    OpsMetrics,
)

logger = logging.getLogger("services.mlflow")


class MLflowService:
    """
    Service for logging evaluation metrics to MLflow.
    
    Handles experiment setup, run management, and metric logging.
    """
    
    def __init__(
        self,
        experiment_name: str = "medical-simplification-evals",
        tracking_uri: str | None = None,
    ):
        """
        Initialize MLflow service.
        
        Args:
            experiment_name: Name of the MLflow experiment.
            tracking_uri: MLflow tracking server URI. Uses local if None.
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self._mlflow = None
        self._experiment_id = None
    
    def _get_mlflow(self):
        """Lazy load MLflow."""
        if self._mlflow is not None:
            return self._mlflow
        
        try:
            import mlflow
            
            if self.tracking_uri:
                mlflow.set_tracking_uri(self.tracking_uri)
            
            # Create or get experiment
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                self._experiment_id = mlflow.create_experiment(self.experiment_name)
            else:
                self._experiment_id = experiment.experiment_id
            
            mlflow.set_experiment(self.experiment_name)
            
            self._mlflow = mlflow
            logger.info(f"MLflow initialized with experiment: {self.experiment_name}")
            return self._mlflow
        except ImportError:
            logger.warning("MLflow not installed. Run: pip install mlflow")
            return None
        except Exception as e:
            logger.warning(f"Failed to initialize MLflow: {e}")
            return None
    
    def start_run(
        self,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> str | None:
        """
        Start a new MLflow run.
        
        Returns:
            Run ID or None if MLflow is not available.
        """
        mlflow = self._get_mlflow()
        if not mlflow:
            return None
        
        run_name = run_name or f"eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        run = mlflow.start_run(run_name=run_name, tags=tags)
        
        logger.info(f"Started MLflow run: {run.info.run_id}")
        return run.info.run_id
    
    def end_run(self):
        """End the current MLflow run."""
        mlflow = self._get_mlflow()
        if mlflow:
            mlflow.end_run()
    
    def log_params(self, params: dict[str, Any]):
        """Log parameters to the current run."""
        mlflow = self._get_mlflow()
        if not mlflow:
            return
        
        for key, value in params.items():
            # Convert non-string values
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            mlflow.log_param(key, value)
    
    def log_metric(self, key: str, value: float, step: int | None = None):
        """Log a single metric."""
        mlflow = self._get_mlflow()
        if mlflow:
            mlflow.log_metric(key, value, step=step)
    
    def log_metrics(self, metrics: dict[str, float], step: int | None = None):
        """Log multiple metrics."""
        mlflow = self._get_mlflow()
        if mlflow:
            mlflow.log_metrics(metrics, step=step)
    
    def log_artifact(self, local_path: str, artifact_path: str | None = None):
        """Log a file as an artifact."""
        mlflow = self._get_mlflow()
        if mlflow:
            mlflow.log_artifact(local_path, artifact_path)
    
    def log_text(self, text: str, artifact_file: str):
        """Log text content as an artifact."""
        mlflow = self._get_mlflow()
        if mlflow:
            mlflow.log_text(text, artifact_file)
    
    def log_dict(self, dictionary: dict, artifact_file: str):
        """Log a dictionary as a JSON artifact."""
        mlflow = self._get_mlflow()
        if mlflow:
            mlflow.log_dict(dictionary, artifact_file)
    
    def log_readability_metrics(
        self,
        metrics: ReadabilityMetrics,
        prefix: str = "readability",
    ):
        """Log readability metrics."""
        self.log_metrics({
            f"{prefix}/fkgl": metrics.fkgl,
            f"{prefix}/smog": metrics.smog,
            f"{prefix}/avg_sentence_length": metrics.avg_sentence_length,
            f"{prefix}/jargon_density": metrics.jargon_density,
            f"{prefix}/total_tokens": float(metrics.total_tokens),
            f"{prefix}/cui_count": float(metrics.cui_count),
        })
    
    def log_semantic_metrics(
        self,
        metrics: SemanticSimilarityMetrics,
        prefix: str = "semantic",
    ):
        """Log semantic similarity metrics."""
        self.log_metrics({
            f"{prefix}/precision": metrics.precision,
            f"{prefix}/recall": metrics.recall,
            f"{prefix}/f1": metrics.f1,
        })
        self.log_params({f"{prefix}/model": metrics.model_name})
    
    def log_concept_metrics(
        self,
        metrics: ConceptOverlapMetrics,
        prefix: str = "concept",
    ):
        """Log concept overlap metrics."""
        self.log_metrics({
            f"{prefix}/cui_precision": metrics.cui_precision,
            f"{prefix}/cui_recall": metrics.cui_recall,
            f"{prefix}/cui_f1": metrics.cui_f1,
            f"{prefix}/original_cui_count": float(metrics.original_cui_count),
            f"{prefix}/simplified_cui_count": float(metrics.simplified_cui_count),
            f"{prefix}/shared_cui_count": float(metrics.shared_cui_count),
        })
        if metrics.semantic_type_weighted_f1 is not None:
            self.log_metric(f"{prefix}/weighted_f1", metrics.semantic_type_weighted_f1)
    
    def log_nli_metrics(
        self,
        metrics: NLIDistributionMetrics,
        prefix: str = "nli",
    ):
        """Log NLI distribution metrics."""
        self.log_metrics({
            f"{prefix}/entailment_rate": metrics.entailment_rate,
            f"{prefix}/neutral_rate": metrics.neutral_rate,
            f"{prefix}/contradiction_rate": metrics.contradiction_rate,
            f"{prefix}/total_pairs": float(metrics.total_pairs),
            f"{prefix}/avg_entailment_confidence": metrics.avg_entailment_confidence,
            f"{prefix}/avg_contradiction_confidence": metrics.avg_contradiction_confidence,
        })
    
    def log_coverage_metrics(
        self,
        metrics: CoverageMetrics,
        prefix: str = "coverage",
    ):
        """Log coverage metrics."""
        self.log_metrics({
            f"{prefix}/reference_cui_recall": metrics.reference_cui_recall,
            f"{prefix}/key_measurements_present": metrics.key_measurements_present,
            f"{prefix}/total_reference_cuis": float(metrics.total_reference_cuis),
            f"{prefix}/covered_reference_cuis": float(metrics.covered_reference_cuis),
            f"{prefix}/total_measurements": float(metrics.total_measurements),
            f"{prefix}/preserved_measurements": float(metrics.preserved_measurements),
        })
    
    def log_ops_metrics(
        self,
        metrics: OpsMetrics,
        prefix: str = "ops",
    ):
        """Log operations metrics."""
        self.log_metrics({
            f"{prefix}/latency_p50_ms": metrics.latency_p50_ms,
            f"{prefix}/latency_p95_ms": metrics.latency_p95_ms,
            f"{prefix}/latency_mean_ms": metrics.latency_mean_ms,
            f"{prefix}/cost_per_report_tokens": metrics.cost_per_report_tokens,
            f"{prefix}/retry_rate": metrics.retry_rate,
            f"{prefix}/refinement_rate": metrics.refinement_rate,
            f"{prefix}/first_pass_success_rate": metrics.first_pass_success_rate,
        })
    
    def log_dataset_evaluation(self, evaluation: DatasetEvaluation):
        """Log a complete dataset evaluation."""
        # Log basic params
        self.log_params({
            "dataset_name": evaluation.dataset_name,
            "model_version": evaluation.model_version,
            "sample_count": evaluation.sample_count,
        })
        
        # Log all metrics
        if evaluation.readability_mean:
            self.log_readability_metrics(evaluation.readability_mean)
        
        if evaluation.semantic_mean:
            self.log_semantic_metrics(evaluation.semantic_mean)
        
        if evaluation.concept_mean:
            self.log_concept_metrics(evaluation.concept_mean)
        
        if evaluation.nli_distribution:
            self.log_nli_metrics(evaluation.nli_distribution)
        
        if evaluation.coverage_mean:
            self.log_coverage_metrics(evaluation.coverage_mean)
        
        if evaluation.ops:
            self.log_ops_metrics(evaluation.ops)
        
        # Log full results as artifact
        self.log_dict(
            evaluation.model_dump(mode="json"),
            "evaluation_results.json",
        )


# Singleton instance
_mlflow_service: MLflowService | None = None


def get_mlflow_service(
    experiment_name: str = "medical-simplification-evals",
    tracking_uri: str | None = None,
) -> MLflowService:
    """Get or create the MLflow service singleton."""
    global _mlflow_service
    
    if _mlflow_service is None:
        _mlflow_service = MLflowService(
            experiment_name=experiment_name,
            tracking_uri=tracking_uri,
        )
    
    return _mlflow_service
