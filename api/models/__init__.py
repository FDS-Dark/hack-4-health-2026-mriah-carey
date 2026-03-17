"""Data models for medical report parsing."""

from models.report_schema import ParsedReport, MainIdea, SupportingDetail
from models.simplified_schema import SimplifiedReport, SimplifiedMainIdea, SimplifiedDetail
from models.api_schema import ProcessingResponse
from models.validation import (
    ErrorCode,
    Severity,
    ValidationError,
    ValidationResult,
    PipelineValidationResult,
)

__all__ = [
    "ParsedReport", "MainIdea", "SupportingDetail",
    "SimplifiedReport", "SimplifiedMainIdea", "SimplifiedDetail",
    "ProcessingResponse",
    "ErrorCode", "Severity", "ValidationError", "ValidationResult", "PipelineValidationResult",
]
