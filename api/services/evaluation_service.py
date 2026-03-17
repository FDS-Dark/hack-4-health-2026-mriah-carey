"""Evaluation orchestrator service."""

import logging
from dataclasses import dataclass, field
from typing import Any

from models.evaluation import (
    DatasetEvaluation,
    SampleEvaluation,
    ReadabilityMetrics,
    ReadabilityComparison,
    SimplificationQualityMetrics,
    SemanticSimilarityMetrics,
    ConceptOverlapMetrics,
    NLIDistributionMetrics,
    CoverageMetrics,
    OpsMetrics,
)
from evaluators.base import BaseEvaluator
from evaluators.readability import ReadabilityEvaluator
from evaluators.simplification import SimplificationEvaluator
from evaluators.semantic import SemanticSimilarityEvaluator
from evaluators.concept import ConceptOverlapEvaluator
from evaluators.nli import NLIDistributionEvaluator
from evaluators.coverage import CoverageEvaluator
from evaluators.ops import OpsEvaluator
from services.mlflow_service import MLflowService, get_mlflow_service

logger = logging.getLogger("services.evaluation")


@dataclass
class EvaluationConfig:
    """Configuration for evaluation runs."""
    # Enable/disable evaluators
    enable_readability: bool = True
    enable_simplification: bool = True  # Before/after comparison + quality metrics
    enable_semantic: bool = True
    enable_concept: bool = True
    enable_nli: bool = True
    enable_coverage: bool = True
    enable_ops: bool = True
    
    # Model choices
    semantic_model: str = "pubmedbert"  # or "bioclinicalbert"
    nli_model: str = "cross-encoder/nli-deberta-v3-base"
    
    # MLflow settings
    experiment_name: str = "medical-simplification-evals"
    tracking_uri: str | None = None
    
    # Dataset info (for logging)
    dataset_name: str = "unknown"
    model_version: str = "v1"


class EvaluationOrchestrator:
    """
    Orchestrates evaluation across all metrics.
    
    Runs configured evaluators and logs results to MLflow.
    """
    
    def __init__(
        self,
        config: EvaluationConfig | None = None,
        opensearch_client=None,
    ):
        """
        Initialize the evaluation orchestrator.
        
        Args:
            config: Evaluation configuration.
            opensearch_client: OpenSearch client for UMLS lookups.
        """
        self.config = config or EvaluationConfig()
        self.opensearch_client = opensearch_client
        
        # Initialize evaluators
        self.evaluators: dict[str, BaseEvaluator] = {}
        self._init_evaluators()
        
        # MLflow service
        self.mlflow = get_mlflow_service(
            experiment_name=self.config.experiment_name,
            tracking_uri=self.config.tracking_uri,
        )
        
        logger.info(f"Initialized EvaluationOrchestrator with {len(self.evaluators)} evaluators")
    
    def _init_evaluators(self):
        """Initialize configured evaluators."""
        if self.config.enable_readability:
            self.evaluators["readability"] = ReadabilityEvaluator(
                umls_client=self.opensearch_client
            )
            logger.info("  ✓ ReadabilityEvaluator enabled")
        
        if self.config.enable_simplification:
            self.evaluators["simplification"] = SimplificationEvaluator(
                opensearch_client=self.opensearch_client
            )
            logger.info("  ✓ SimplificationEvaluator enabled (before/after + quality)")
        
        if self.config.enable_semantic:
            self.evaluators["semantic"] = SemanticSimilarityEvaluator(
                model_name=self.config.semantic_model
            )
            logger.info("  ✓ SemanticSimilarityEvaluator enabled")
        
        if self.config.enable_concept:
            self.evaluators["concept"] = ConceptOverlapEvaluator(
                opensearch_client=self.opensearch_client
            )
            logger.info("  ✓ ConceptOverlapEvaluator enabled")
        
        if self.config.enable_nli:
            self.evaluators["nli"] = NLIDistributionEvaluator(
                model_name=self.config.nli_model
            )
            logger.info("  ✓ NLIDistributionEvaluator enabled")
        
        if self.config.enable_coverage:
            self.evaluators["coverage"] = CoverageEvaluator(
                opensearch_client=self.opensearch_client
            )
            logger.info("  ✓ CoverageEvaluator enabled")
        
        if self.config.enable_ops:
            self.evaluators["ops"] = OpsEvaluator()
            logger.info("  ✓ OpsEvaluator enabled")
    
    def evaluate_sample(
        self,
        original_text: str,
        simplified_text: str,
        reference_text: str | None = None,
        sample_id: str = "sample",
    ) -> SampleEvaluation:
        """
        Evaluate a single sample with all enabled evaluators.
        
        Args:
            original_text: Original medical text.
            simplified_text: Simplified text to evaluate.
            reference_text: Optional reference simplification.
            sample_id: Unique identifier for this sample.
        
        Returns:
            SampleEvaluation with all computed metrics.
        """
        logger.info(f"Evaluating sample: {sample_id}")
        
        result = SampleEvaluation(
            sample_id=sample_id,
            original_text=original_text,
            simplified_text=simplified_text,
            reference_text=reference_text,
        )
        
        # Run each evaluator
        if "readability" in self.evaluators:
            result.readability = self.evaluators["readability"].evaluate_sample(
                original_text, simplified_text, reference_text
            )
        
        if "simplification" in self.evaluators:
            readability_comp, quality = self.evaluators["simplification"].evaluate_sample(
                original_text, simplified_text, reference_text
            )
            result.readability_comparison = readability_comp
            result.simplification_quality = quality
        
        if "semantic" in self.evaluators:
            result.semantic = self.evaluators["semantic"].evaluate_sample(
                original_text, simplified_text, reference_text
            )
        
        if "concept" in self.evaluators:
            result.concept = self.evaluators["concept"].evaluate_sample(
                original_text, simplified_text, reference_text
            )
        
        if "nli" in self.evaluators:
            result.nli = self.evaluators["nli"].evaluate_sample(
                original_text, simplified_text, reference_text
            )
        
        if "coverage" in self.evaluators:
            result.coverage = self.evaluators["coverage"].evaluate_sample(
                original_text, simplified_text, reference_text
            )
        
        return result
    
    def evaluate_dataset(
        self,
        samples: list[dict],
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> DatasetEvaluation:
        """
        Evaluate a dataset of samples.
        
        Args:
            samples: List of dicts with 'original_text', 'simplified_text',
                     optionally 'reference_text' and 'sample_id'.
            run_name: Optional name for the MLflow run.
            tags: Optional tags for the MLflow run.
        
        Returns:
            DatasetEvaluation with aggregated metrics.
        """
        logger.info(f"Evaluating dataset: {len(samples)} samples")
        
        # Start MLflow run
        run_id = self.mlflow.start_run(run_name=run_name, tags=tags)
        
        try:
            # Evaluate all samples
            sample_results: list[SampleEvaluation] = []
            
            for i, sample in enumerate(samples):
                sample_id = sample.get("sample_id", f"sample_{i}")
                
                result = self.evaluate_sample(
                    original_text=sample["original_text"],
                    simplified_text=sample["simplified_text"],
                    reference_text=sample.get("reference_text"),
                    sample_id=sample_id,
                )
                sample_results.append(result)
                
                logger.info(f"  Completed {i+1}/{len(samples)}")
            
            # Aggregate results
            dataset_eval = self._aggregate_results(sample_results, run_id or "")
            
            # Log to MLflow
            self.mlflow.log_dataset_evaluation(dataset_eval)
            
            return dataset_eval
        
        finally:
            self.mlflow.end_run()
    
    def _aggregate_results(
        self,
        samples: list[SampleEvaluation],
        run_id: str,
    ) -> DatasetEvaluation:
        """Aggregate sample results into dataset-level metrics."""
        n = len(samples)
        
        # Aggregate readability
        readability_mean = None
        if any(s.readability for s in samples):
            valid = [s.readability for s in samples if s.readability]
            readability_mean = ReadabilityMetrics(
                fkgl=round(sum(r.fkgl for r in valid) / len(valid), 2),
                smog=round(sum(r.smog for r in valid) / len(valid), 2),
                avg_sentence_length=round(sum(r.avg_sentence_length for r in valid) / len(valid), 2),
                jargon_density=round(sum(r.jargon_density for r in valid) / len(valid), 4),
                total_tokens=sum(r.total_tokens for r in valid),
                cui_count=sum(r.cui_count for r in valid),
            )
        
        # Aggregate readability comparison (before/after)
        readability_comparison_mean = None
        if any(s.readability_comparison for s in samples):
            valid = [s.readability_comparison for s in samples if s.readability_comparison]
            readability_comparison_mean = ReadabilityComparison(
                original_fkgl=round(sum(r.original_fkgl for r in valid) / len(valid), 1),
                original_smog=round(sum(r.original_smog for r in valid) / len(valid), 1),
                original_avg_sentence_length=round(sum(r.original_avg_sentence_length for r in valid) / len(valid), 1),
                original_jargon_density=round(sum(r.original_jargon_density for r in valid) / len(valid), 4),
                simplified_fkgl=round(sum(r.simplified_fkgl for r in valid) / len(valid), 1),
                simplified_smog=round(sum(r.simplified_smog for r in valid) / len(valid), 1),
                simplified_avg_sentence_length=round(sum(r.simplified_avg_sentence_length for r in valid) / len(valid), 1),
                simplified_jargon_density=round(sum(r.simplified_jargon_density for r in valid) / len(valid), 4),
                fkgl_delta=round(sum(r.fkgl_delta for r in valid) / len(valid), 1),
                smog_delta=round(sum(r.smog_delta for r in valid) / len(valid), 1),
                sentence_length_delta=round(sum(r.sentence_length_delta for r in valid) / len(valid), 1),
                jargon_reduction=round(sum(r.jargon_reduction for r in valid) / len(valid), 4),
            )
        
        # Aggregate simplification quality
        simplification_quality_mean = None
        if any(s.simplification_quality for s in samples):
            valid = [s.simplification_quality for s in samples if s.simplification_quality]
            # Collect all missing items
            all_missing_diag = []
            all_missing_meds = []
            all_missing_meas = []
            for q in valid:
                all_missing_diag.extend(q.missing_diagnoses)
                all_missing_meds.extend(q.missing_medications)
                all_missing_meas.extend(q.missing_measurements)
            
            simplification_quality_mean = SimplificationQualityMetrics(
                concept_recall=round(sum(q.concept_recall for q in valid) / len(valid), 4),
                critical_concept_recall=round(sum(q.critical_concept_recall for q in valid) / len(valid), 4),
                measurement_preservation=round(sum(q.measurement_preservation for q in valid) / len(valid), 4),
                original_words_per_concept=round(sum(q.original_words_per_concept for q in valid) / len(valid), 2),
                simplified_words_per_concept=round(sum(q.simplified_words_per_concept for q in valid) / len(valid), 2),
                expansion_ratio=round(sum(q.expansion_ratio for q in valid) / len(valid), 2),
                missing_diagnoses=list(set(all_missing_diag))[:10],
                missing_medications=list(set(all_missing_meds))[:10],
                missing_measurements=list(set(all_missing_meas))[:10],
                simplification_score=round(sum(q.simplification_score for q in valid) / len(valid), 4),
            )
        
        # Aggregate semantic
        semantic_mean = None
        if any(s.semantic for s in samples):
            valid = [s.semantic for s in samples if s.semantic]
            semantic_mean = SemanticSimilarityMetrics(
                precision=round(sum(r.precision for r in valid) / len(valid), 4),
                recall=round(sum(r.recall for r in valid) / len(valid), 4),
                f1=round(sum(r.f1 for r in valid) / len(valid), 4),
                model_name=valid[0].model_name if valid else "",
            )
        
        # Aggregate concept
        concept_mean = None
        if any(s.concept for s in samples):
            valid = [s.concept for s in samples if s.concept]
            concept_mean = ConceptOverlapMetrics(
                cui_precision=round(sum(r.cui_precision for r in valid) / len(valid), 4),
                cui_recall=round(sum(r.cui_recall for r in valid) / len(valid), 4),
                cui_f1=round(sum(r.cui_f1 for r in valid) / len(valid), 4),
                original_cui_count=sum(r.original_cui_count for r in valid),
                simplified_cui_count=sum(r.simplified_cui_count for r in valid),
                shared_cui_count=sum(r.shared_cui_count for r in valid),
                semantic_type_weighted_f1=(
                    round(sum(r.semantic_type_weighted_f1 for r in valid if r.semantic_type_weighted_f1) / len(valid), 4)
                    if any(r.semantic_type_weighted_f1 for r in valid) else None
                ),
            )
        
        # Aggregate NLI (special: sum counts, then compute rates)
        nli_distribution = None
        if any(s.nli for s in samples):
            valid = [s.nli for s in samples if s.nli]
            total_pairs = sum(r.total_pairs for r in valid)
            if total_pairs > 0:
                # Weight rates by number of pairs
                entailment_sum = sum(r.entailment_rate * r.total_pairs for r in valid)
                neutral_sum = sum(r.neutral_rate * r.total_pairs for r in valid)
                contradiction_sum = sum(r.contradiction_rate * r.total_pairs for r in valid)
                
                nli_distribution = NLIDistributionMetrics(
                    entailment_rate=round(entailment_sum / total_pairs, 4),
                    neutral_rate=round(neutral_sum / total_pairs, 4),
                    contradiction_rate=round(contradiction_sum / total_pairs, 4),
                    total_pairs=total_pairs,
                    avg_entailment_confidence=round(
                        sum(r.avg_entailment_confidence for r in valid) / len(valid), 4
                    ),
                    avg_contradiction_confidence=round(
                        sum(r.avg_contradiction_confidence for r in valid) / len(valid), 4
                    ),
                )
        
        # Aggregate coverage
        coverage_mean = None
        if any(s.coverage for s in samples):
            valid = [s.coverage for s in samples if s.coverage]
            coverage_mean = CoverageMetrics(
                reference_cui_recall=round(sum(r.reference_cui_recall for r in valid) / len(valid), 4),
                key_measurements_present=round(sum(r.key_measurements_present for r in valid) / len(valid), 4),
                total_reference_cuis=sum(r.total_reference_cuis for r in valid),
                covered_reference_cuis=sum(r.covered_reference_cuis for r in valid),
                total_measurements=sum(r.total_measurements for r in valid),
                preserved_measurements=sum(r.preserved_measurements for r in valid),
            )
        
        # Get ops metrics from evaluator
        ops = None
        if "ops" in self.evaluators:
            ops_evaluator: OpsEvaluator = self.evaluators["ops"]
            ops = ops_evaluator.get_metrics()
        
        return DatasetEvaluation(
            run_id=run_id,
            dataset_name=self.config.dataset_name,
            model_version=self.config.model_version,
            sample_count=n,
            readability_mean=readability_mean,
            readability_comparison_mean=readability_comparison_mean,
            simplification_quality_mean=simplification_quality_mean,
            semantic_mean=semantic_mean,
            concept_mean=concept_mean,
            nli_distribution=nli_distribution,
            coverage_mean=coverage_mean,
            ops=ops,
            samples=samples,
        )


def create_evaluation_orchestrator(
    config: EvaluationConfig | None = None,
    opensearch_client=None,
) -> EvaluationOrchestrator:
    """Factory function to create an evaluation orchestrator."""
    return EvaluationOrchestrator(
        config=config,
        opensearch_client=opensearch_client,
    )
