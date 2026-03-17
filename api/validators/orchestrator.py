"""Validation orchestrator to run all validators."""

import logging

from clients.opensearch import OpenSearchClient
from models.validation import PipelineValidationResult, ValidationResult
from validators.base import BaseValidator
from validators.numeric import NumericValidator
from validators.evidence_span import EvidenceSpanValidator
from validators.umls_grounding import UMLSGroundingValidator
from validators.contradiction_nli import ContradictionNLIValidator
from validators.recommendation_policy import RecommendationPolicyValidator
from validators.empty_chunk import EmptyChunkValidator
from validators.concept_safety import ConceptSafetyValidator

logger = logging.getLogger("validators.orchestrator")


class ValidationOrchestrator:
    """
    Orchestrates all validation checks for the pipeline.
    
    Runs validators in order and collects results.
    """
    
    def __init__(
        self,
        opensearch_client: OpenSearchClient | None = None,
        enable_empty_chunk: bool = True,
        enable_numeric: bool = True,
        enable_evidence_span: bool = False,
        enable_umls_grounding: bool = True,
        enable_contradiction_nli: bool = True,
        enable_recommendation_policy: bool = True,
        enable_concept_safety: bool = True,
        concept_safety_threshold: float = 0.70,
    ):
        logger.info("Initializing ValidationOrchestrator")
        
        self.validators: list[BaseValidator] = []
        self._concept_safety_validator: ConceptSafetyValidator | None = None
        
        # Empty chunk validator runs FIRST to skip invalid chunks early
        if enable_empty_chunk:
            logger.info("  ✓ EmptyChunkValidator enabled")
            self.validators.append(EmptyChunkValidator())
        
        if enable_numeric:
            logger.info("  ✓ NumericValidator enabled")
            self.validators.append(NumericValidator())
        
        if enable_evidence_span:
            logger.info("  ✓ EvidenceSpanValidator enabled")
            self.validators.append(EvidenceSpanValidator())
        
        if enable_umls_grounding:
            logger.info("  ✓ UMLSGroundingValidator enabled")
            self.validators.append(UMLSGroundingValidator(
                opensearch_client=opensearch_client
            ))
        
        if enable_contradiction_nli:
            logger.info("  ✓ ContradictionNLIValidator enabled")
            self.validators.append(ContradictionNLIValidator())
        
        if enable_recommendation_policy:
            logger.info("  ✓ RecommendationPolicyValidator enabled")
            self.validators.append(RecommendationPolicyValidator())
        
        # Concept safety validator - runs separately for flagging
        if enable_concept_safety:
            logger.info("  ✓ ConceptSafetyValidator enabled")
            self._concept_safety_validator = ConceptSafetyValidator(
                min_recall_threshold=concept_safety_threshold,
            )
        
        logger.info(f"Total validators: {len(self.validators)}")
    
    def add_validator(self, validator: BaseValidator) -> None:
        """Add a custom validator to the pipeline."""
        logger.info(f"Adding custom validator: {validator.name}")
        self.validators.append(validator)
    
    def validate(
        self,
        original_text: str,
        simplified_text: str,
        iteration: int = 1
    ) -> PipelineValidationResult:
        """Run all validators and return combined results."""
        logger.info(f"\n{'#'*70}")
        logger.info(f"# VALIDATION ORCHESTRATOR - Iteration {iteration}")
        logger.info(f"{'#'*70}")
        logger.info(f"Original text length: {len(original_text)} chars")
        logger.info(f"Simplified text length: {len(simplified_text)} chars")
        logger.info(f"Running {len(self.validators)} validators...")
        
        results: list[ValidationResult] = []
        all_passed = True
        
        for i, validator in enumerate(self.validators):
            logger.info(f"\n[{i+1}/{len(self.validators)}] Running {validator.name}...")
            
            result = validator.validate(original_text, simplified_text)
            results.append(result)
            
            if result.passed:
                logger.info(f"  → PASSED ✓")
            else:
                logger.warning(f"  → FAILED with {len(result.errors)} errors")
                for err in result.errors:
                    logger.warning(f"    - [{err.code.value}] {err.message[:60]}...")
                
                if validator.is_hard_gate:
                    all_passed = False
        
        # Run concept safety check (separate from hard gates)
        needs_review = False
        concept_recall = None
        
        if self._concept_safety_validator:
            logger.info(f"\n[SAFETY] Running ConceptSafetyValidator...")
            safety_result = self._concept_safety_validator.validate(
                original_text, simplified_text
            )
            concept_recall = safety_result.recall
            
            if safety_result.needs_review:
                logger.warning(
                    f"  → LOW CONCEPT RECALL: {safety_result.recall:.1%} "
                    f"({len(safety_result.missing_concepts)} concepts missing)"
                )
                needs_review = True
            elif not safety_result.is_safe:
                logger.warning(
                    f"  → CONCEPT RECALL WARNING: {safety_result.recall:.1%}"
                )
            else:
                logger.info(f"  → PASSED ✓ ({safety_result.recall:.1%} recall)")
        
        # Summary
        logger.info(f"\n{'='*70}")
        logger.info(f"VALIDATION SUMMARY - Iteration {iteration}")
        logger.info(f"{'='*70}")
        
        passed_count = sum(1 for r in results if r.passed)
        failed_count = len(results) - passed_count
        total_errors = sum(len(r.errors) for r in results)
        
        logger.info(f"Validators passed: {passed_count}/{len(results)}")
        logger.info(f"Validators failed: {failed_count}/{len(results)}")
        logger.info(f"Total errors: {total_errors}")
        if concept_recall is not None:
            logger.info(f"Concept recall: {concept_recall:.1%}")
        logger.info(f"Needs review: {'YES' if needs_review else 'NO'}")
        logger.info(f"Overall result: {'PASSED ✓' if all_passed else 'FAILED ✗'}")
        
        return PipelineValidationResult(
            passed=all_passed,
            results=results,
            iteration=iteration,
            needs_review=needs_review
        )
    
    def validate_with_retry(
        self,
        original_text: str,
        simplified_text: str,
        max_iterations: int = 3
    ) -> tuple[PipelineValidationResult, bool]:
        """Run validation, returning whether repair should be attempted."""
        result = self.validate(original_text, simplified_text)
        
        if result.passed:
            return result, False
        
        if result.iteration >= max_iterations:
            logger.warning(f"Max iterations ({max_iterations}) reached - marking for review")
            result.needs_review = True
            return result, False
        
        return result, True


def create_default_orchestrator(
    opensearch_client: OpenSearchClient | None = None
) -> ValidationOrchestrator:
    """Create a ValidationOrchestrator with default configuration."""
    return ValidationOrchestrator(
        opensearch_client=opensearch_client,
        enable_empty_chunk=True,
        enable_numeric=True,
        enable_evidence_span=False,
        enable_umls_grounding=True,
        enable_contradiction_nli=True,
        enable_recommendation_policy=True,
    )


def create_fast_orchestrator() -> ValidationOrchestrator:
    """Create a fast orchestrator with only cheap validators."""
    return ValidationOrchestrator(
        enable_empty_chunk=True,
        enable_numeric=True,
        enable_evidence_span=False,
        enable_umls_grounding=False,
        enable_contradiction_nli=False,
        enable_recommendation_policy=True,
    )
