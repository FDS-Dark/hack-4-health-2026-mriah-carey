"""Validation models for pipeline validation results."""

from enum import Enum
from pydantic import BaseModel, Field


class Severity(str, Enum):
    """Severity level for validation errors."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ErrorCode(str, Enum):
    """Standardized error codes for validation failures."""
    # V1: Numeric + unit preservation
    NUMERIC_MISMATCH = "NUMERIC_MISMATCH"
    NUMERIC_MISSING = "NUMERIC_MISSING"
    NUMERIC_ADDED = "NUMERIC_ADDED"
    UNIT_MISMATCH = "UNIT_MISMATCH"
    LATERALITY_FLIP = "LATERALITY_FLIP"
    
    # V2: Evidence-span validation
    EVIDENCE_NOT_FOUND = "EVIDENCE_NOT_FOUND"
    CLAIM_WITHOUT_EVIDENCE = "CLAIM_WITHOUT_EVIDENCE"
    
    # V3: UMLS grounding precision
    UMLS_CUI_INTRODUCED = "UMLS_CUI_INTRODUCED"
    UMLS_HIGH_RISK_TERM = "UMLS_HIGH_RISK_TERM"
    UMLS_PRECISION_LOW = "UMLS_PRECISION_LOW"
    
    # V4: Contradiction NLI
    CONTRADICTION_DETECTED = "CONTRADICTION_DETECTED"
    
    # V5: Recommendation policy
    UNSAFE_RECOMMENDATION = "UNSAFE_RECOMMENDATION"
    UNGROUNDED_ADVICE = "UNGROUNDED_ADVICE"
    
    # Empty/Meta-talk detection
    EMPTY_CONTENT = "EMPTY_CONTENT"
    META_TALK_DETECTED = "META_TALK_DETECTED"
    
    # Concept safety
    CONCEPT_MISSING = "CONCEPT_MISSING"
    CONCEPT_RECALL_LOW = "CONCEPT_RECALL_LOW"


class ValidationError(BaseModel):
    """A single validation error with context for repair."""
    code: ErrorCode = Field(..., description="Error code identifying the type of issue")
    severity: Severity = Field(..., description="Severity level of the error")
    message: str = Field(..., description="Human-readable error description")
    summary_span: str | None = Field(None, description="The problematic text span in the summary")
    source_evidence: str | None = Field(None, description="Relevant text from the original source")
    fix: str | None = Field(None, description="Suggested fix for the error")
    metadata: dict | None = Field(None, description="Additional context (CUIs, sentence indices, etc.)")


class ValidationResult(BaseModel):
    """Result of running validation checks."""
    passed: bool = Field(..., description="Whether all validations passed")
    validator_name: str = Field(..., description="Name of the validator that produced this result")
    errors: list[ValidationError] = Field(default_factory=list, description="List of validation errors")
    
    @classmethod
    def success(cls, validator_name: str) -> "ValidationResult":
        """Create a successful validation result."""
        return cls(passed=True, validator_name=validator_name, errors=[])
    
    @classmethod
    def failure(cls, validator_name: str, errors: list[ValidationError]) -> "ValidationResult":
        """Create a failed validation result."""
        return cls(passed=False, validator_name=validator_name, errors=errors)


class PipelineValidationResult(BaseModel):
    """Combined result of all pipeline validations."""
    passed: bool = Field(..., description="Whether all validations passed")
    results: list[ValidationResult] = Field(default_factory=list, description="Results from each validator")
    iteration: int = Field(default=1, description="Current repair iteration")
    needs_review: bool = Field(default=False, description="Whether manual review is required")
    
    @property
    def all_errors(self) -> list[ValidationError]:
        """Get all errors across all validators."""
        errors = []
        for result in self.results:
            errors.extend(result.errors)
        return errors
    
    @property
    def hard_failures(self) -> list[ValidationError]:
        """Get only HIGH and CRITICAL severity errors."""
        return [e for e in self.all_errors if e.severity in (Severity.HIGH, Severity.CRITICAL)]
    
    def to_repair_prompt(self) -> str:
        """Generate a prompt for Gemini to repair the issues."""
        if not self.all_errors:
            return ""
        
        lines = ["The simplified text has the following validation errors that must be fixed:\n"]
        
        for i, error in enumerate(self.all_errors, 1):
            lines.append(f"{i}. [{error.code.value}] {error.message}")
            if error.summary_span:
                lines.append(f"   Problem text: \"{error.summary_span}\"")
            if error.source_evidence:
                lines.append(f"   Original says: \"{error.source_evidence}\"")
            if error.fix:
                lines.append(f"   Suggested fix: {error.fix}")
            lines.append("")
        
        return "\n".join(lines)
