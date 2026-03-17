"""Evaluation models for MLflow-based evaluation metrics."""

from enum import Enum
from pydantic import BaseModel, Field


class EvaluationCategory(str, Enum):
    """Categories of evaluation metrics."""
    READABILITY = "readability"
    SIMPLIFICATION = "simplification"
    SEMANTIC = "semantic"
    CONCEPT = "concept"
    NLI = "nli"
    COVERAGE = "coverage"
    OPS = "ops"


# =============================================================================
# E1: Readability & Accessibility Metrics
# =============================================================================

class ReadabilityMetrics(BaseModel):
    """Readability scores for a single document."""
    fkgl: float = Field(..., description="Flesch-Kincaid Grade Level")
    smog: float = Field(..., description="SMOG Index")
    avg_sentence_length: float = Field(..., description="Average sentence length in words")
    jargon_density: float = Field(..., description="Ratio of UMLS CUIs to total tokens")
    total_tokens: int = Field(..., description="Total token count")
    cui_count: int = Field(..., description="Number of UMLS CUIs detected")


class ReadabilityComparison(BaseModel):
    """Before/after readability comparison for simplification."""
    # Original (before) metrics
    original_fkgl: float = Field(..., description="Original FKGL grade level")
    original_smog: float = Field(..., description="Original SMOG index")
    original_avg_sentence_length: float = Field(..., description="Original avg sentence length")
    original_jargon_density: float = Field(..., description="Original jargon density")
    
    # Simplified (after) metrics
    simplified_fkgl: float = Field(..., description="Simplified FKGL grade level")
    simplified_smog: float = Field(..., description="Simplified SMOG index")
    simplified_avg_sentence_length: float = Field(..., description="Simplified avg sentence length")
    simplified_jargon_density: float = Field(..., description="Simplified jargon density")
    
    # Deltas (negative = improvement for grade level)
    fkgl_delta: float = Field(..., description="Change in FKGL (negative = simpler)")
    smog_delta: float = Field(..., description="Change in SMOG (negative = simpler)")
    sentence_length_delta: float = Field(..., description="Change in avg sentence length")
    jargon_reduction: float = Field(..., description="Reduction in jargon density (positive = better)")


# =============================================================================
# E1b: Simplification Quality Metrics
# =============================================================================

class SimplificationQualityMetrics(BaseModel):
    """Metrics measuring quality of simplification."""
    # Concept preservation
    concept_recall: float = Field(..., description="Fraction of original concepts preserved")
    critical_concept_recall: float = Field(..., description="Fraction of critical concepts preserved (diagnoses, meds, measurements)")
    measurement_preservation: float = Field(..., description="Fraction of measurements preserved exactly")
    
    # Information density
    original_words_per_concept: float = Field(..., description="Words per medical concept in original")
    simplified_words_per_concept: float = Field(..., description="Words per medical concept in simplified")
    expansion_ratio: float = Field(..., description="Ratio of simplified to original length")
    
    # Missing critical items
    missing_diagnoses: list[str] = Field(default_factory=list, description="Diagnoses not found in output")
    missing_medications: list[str] = Field(default_factory=list, description="Medications not found in output")
    missing_measurements: list[str] = Field(default_factory=list, description="Measurements not found in output")
    
    # Overall score
    simplification_score: float = Field(..., description="Overall simplification quality score (0-1)")


# =============================================================================
# E2: Semantic Similarity (BERTScore)
# =============================================================================

class SemanticSimilarityMetrics(BaseModel):
    """BERTScore semantic similarity metrics."""
    precision: float = Field(..., description="BERTScore precision")
    recall: float = Field(..., description="BERTScore recall")
    f1: float = Field(..., description="BERTScore F1")
    model_name: str = Field(..., description="Encoder model used")


# =============================================================================
# E3: Concept Overlap (UMLS-based)
# =============================================================================

class ConceptOverlapMetrics(BaseModel):
    """UMLS concept overlap metrics."""
    cui_precision: float = Field(..., description="CUI precision (simplified ∩ original / simplified)")
    cui_recall: float = Field(..., description="CUI recall (simplified ∩ original / original)")
    cui_f1: float = Field(..., description="CUI F1 score")
    original_cui_count: int = Field(..., description="CUIs in original")
    simplified_cui_count: int = Field(..., description="CUIs in simplified")
    shared_cui_count: int = Field(..., description="CUIs in both")
    semantic_type_weighted_f1: float | None = Field(None, description="F1 weighted by semantic type importance")


# =============================================================================
# E4: NLI Distribution
# =============================================================================

class NLIDistributionMetrics(BaseModel):
    """NLI distribution metrics."""
    entailment_rate: float = Field(..., description="Rate of entailment predictions")
    neutral_rate: float = Field(..., description="Rate of neutral predictions")
    contradiction_rate: float = Field(..., description="Rate of contradiction predictions")
    total_pairs: int = Field(..., description="Total sentence pairs evaluated")
    avg_entailment_confidence: float = Field(..., description="Average confidence for entailment")
    avg_contradiction_confidence: float = Field(..., description="Average confidence for contradiction")


# =============================================================================
# E5: Coverage Metrics
# =============================================================================

class CoverageMetrics(BaseModel):
    """Concept and measurement preservation metrics."""
    reference_cui_recall: float = Field(..., description="Recall of original CUIs in simplified output")
    key_measurements_present: float = Field(..., description="Rate of key measurements preserved")
    total_reference_cuis: int = Field(..., description="Total CUIs in original text")
    covered_reference_cuis: int = Field(..., description="Original CUIs found in simplified output")
    total_measurements: int = Field(..., description="Total key measurements in original")
    preserved_measurements: int = Field(..., description="Measurements found in simplified")


# =============================================================================
# E6: Operations Metrics
# =============================================================================

class OpsMetrics(BaseModel):
    """Operational metrics for system performance."""
    latency_p50_ms: float = Field(..., description="50th percentile latency in ms")
    latency_p95_ms: float = Field(..., description="95th percentile latency in ms")
    latency_mean_ms: float = Field(..., description="Mean latency in ms")
    cost_per_report_tokens: float = Field(..., description="Average tokens per report")
    retry_rate: float = Field(..., description="Rate of validation retries")
    refinement_rate: float = Field(..., description="Rate of refinement iterations needed")
    first_pass_success_rate: float = Field(..., description="Rate of first-pass validation success")


# =============================================================================
# Aggregate Results
# =============================================================================

class SampleEvaluation(BaseModel):
    """Evaluation results for a single sample."""
    sample_id: str = Field(..., description="Unique identifier for this sample")
    original_text: str = Field(..., description="Original medical text")
    simplified_text: str = Field(..., description="Simplified text")
    reference_text: str | None = Field(None, description="Reference simplification if available")
    
    # Per-sample metrics
    readability: ReadabilityMetrics | None = None
    readability_comparison: ReadabilityComparison | None = None
    simplification_quality: SimplificationQualityMetrics | None = None
    semantic: SemanticSimilarityMetrics | None = None
    concept: ConceptOverlapMetrics | None = None
    nli: NLIDistributionMetrics | None = None
    coverage: CoverageMetrics | None = None


class DatasetEvaluation(BaseModel):
    """Aggregated evaluation results for a dataset."""
    run_id: str = Field(..., description="MLflow run ID")
    dataset_name: str = Field(..., description="Name of the dataset evaluated")
    model_version: str = Field(..., description="Model/prompt version identifier")
    sample_count: int = Field(..., description="Number of samples evaluated")
    
    # Aggregated metrics
    readability_mean: ReadabilityMetrics | None = None
    readability_comparison_mean: ReadabilityComparison | None = None
    simplification_quality_mean: SimplificationQualityMetrics | None = None
    semantic_mean: SemanticSimilarityMetrics | None = None
    concept_mean: ConceptOverlapMetrics | None = None
    nli_distribution: NLIDistributionMetrics | None = None
    coverage_mean: CoverageMetrics | None = None
    ops: OpsMetrics | None = None
    
    # Individual sample results
    samples: list[SampleEvaluation] = Field(default_factory=list)
