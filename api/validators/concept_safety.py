"""Concept safety validator - flags outputs with low medical concept coverage."""

import logging
import re
from dataclasses import dataclass

from validators.base import BaseValidator
from models.validation import ValidationError, ErrorCode, Severity

logger = logging.getLogger("validators.concept_safety")

# Critical patterns that should be preserved
CRITICAL_PATTERNS = [
    # Diagnoses
    (r'\b(?:diagnosed with|diagnosis of|positive for|confirmed)\s+[\w\s]+', "diagnosis"),
    # Medications with dosages
    (r'\b\d+\s*(?:mg|mcg|g|ml|units?)\b', "medication_dosage"),
    # Lab values
    (r'\b\d+\.?\d*\s*(?:mg/dL|g/dL|mmol/L|mEq/L|U/L|IU/L|ng/mL|pg/mL)\b', "lab_value"),
    # Vital signs
    (r'\b(?:blood pressure|BP|heart rate|HR|temperature|temp|pulse|SpO2|oxygen)\s*:?\s*\d+', "vital_sign"),
    # Laterality
    (r'\b(?:left|right|bilateral)\s+[\w\s]{2,20}', "laterality"),
    # Measurements
    (r'\b\d+\.?\d*\s*(?:mm|cm|m|inches?|feet|kg|lbs?|pounds?)\b', "measurement"),
    # Spinal levels
    (r'\b[CTLS]\d+(?:-[CTLS]?\d+)?\b', "spinal_level"),
    # Percentages
    (r'\b\d+\.?\d*\s*%\b', "percentage"),
]


@dataclass
class ConceptSafetyResult:
    """Result of concept safety validation."""
    is_safe: bool
    recall: float  # Percentage of critical concepts preserved
    missing_concepts: list[str]
    concept_type_coverage: dict[str, tuple[int, int]]  # type -> (found, total)
    needs_review: bool
    errors: list[ValidationError]


class ConceptSafetyValidator(BaseValidator):
    """
    Validates that critical medical concepts are preserved in simplified text.
    
    This is a safety gate - if too many concepts are lost, the output
    is flagged for manual review.
    """
    
    def __init__(
        self,
        min_recall_threshold: float = 0.70,  # Minimum 70% concept recall
        critical_recall_threshold: float = 0.50,  # Below this = needs review
    ):
        """
        Initialize the concept safety validator.
        
        Args:
            min_recall_threshold: Minimum acceptable concept recall (soft gate).
            critical_recall_threshold: Below this threshold, output needs manual review.
        """
        self.min_recall_threshold = min_recall_threshold
        self.critical_recall_threshold = critical_recall_threshold
    
    @property
    def name(self) -> str:
        return "concept_safety"
    
    def _extract_patterns(self, text: str) -> dict[str, set[str]]:
        """Extract all critical patterns from text, grouped by type."""
        by_type: dict[str, set[str]] = {}
        
        for pattern, pattern_type in CRITICAL_PATTERNS:
            if pattern_type not in by_type:
                by_type[pattern_type] = set()
            
            for match in re.finditer(pattern, text, re.IGNORECASE):
                # Normalize the match
                normalized = match.group(0).strip().lower()
                # Remove extra whitespace
                normalized = " ".join(normalized.split())
                by_type[pattern_type].add(normalized)
        
        return by_type
    
    def _calculate_recall(
        self,
        original_patterns: dict[str, set[str]],
        simplified_patterns: dict[str, set[str]],
    ) -> tuple[float, list[str], dict[str, tuple[int, int]]]:
        """
        Calculate concept recall and identify missing concepts.
        
        Returns:
            Tuple of (recall, missing_concepts, coverage_by_type)
        """
        total_original = 0
        total_found = 0
        missing = []
        coverage_by_type: dict[str, tuple[int, int]] = {}
        
        for pattern_type, original_set in original_patterns.items():
            simplified_set = simplified_patterns.get(pattern_type, set())
            
            # Check each original concept
            type_found = 0
            for concept in original_set:
                # Fuzzy match - check if concept appears in any simplified match
                found = False
                
                # Exact match
                if concept in simplified_set:
                    found = True
                else:
                    # Fuzzy match - check if numeric values match
                    # Extract numbers from concept
                    original_nums = set(re.findall(r'\d+\.?\d*', concept))
                    for simp_concept in simplified_set:
                        simp_nums = set(re.findall(r'\d+\.?\d*', simp_concept))
                        if original_nums and original_nums & simp_nums:
                            found = True
                            break
                
                if found:
                    type_found += 1
                else:
                    missing.append(f"[{pattern_type}] {concept}")
            
            total_original += len(original_set)
            total_found += type_found
            coverage_by_type[pattern_type] = (type_found, len(original_set))
        
        recall = total_found / total_original if total_original > 0 else 1.0
        
        return recall, missing, coverage_by_type
    
    def validate(
        self,
        original_text: str,
        simplified_text: str,
    ) -> ConceptSafetyResult:
        """
        Validate that critical medical concepts are preserved.
        
        Args:
            original_text: Original medical text.
            simplified_text: Simplified version to validate.
        
        Returns:
            ConceptSafetyResult with recall metrics and missing concepts.
        """
        # Extract patterns from both
        original_patterns = self._extract_patterns(original_text)
        simplified_patterns = self._extract_patterns(simplified_text)
        
        # Calculate recall
        recall, missing, coverage = self._calculate_recall(
            original_patterns, simplified_patterns
        )
        
        # Build errors
        errors: list[ValidationError] = []
        
        # Check for critically missing concepts
        for missing_concept in missing[:10]:  # Limit to first 10
            errors.append(ValidationError(
                code=ErrorCode.CONCEPT_MISSING,
                severity=Severity.HIGH,
                message=f"Missing critical concept: {missing_concept}",
                source_evidence=missing_concept,
                fix=f"Ensure this concept is mentioned: {missing_concept}",
            ))
        
        is_safe = recall >= self.min_recall_threshold
        needs_review = recall < self.critical_recall_threshold
        
        if needs_review:
            logger.warning(
                f"CONCEPT SAFETY ALERT: Only {recall:.1%} of critical concepts preserved. "
                f"Missing {len(missing)} concepts. Flagging for review."
            )
        
        return ConceptSafetyResult(
            is_safe=is_safe,
            recall=recall,
            missing_concepts=missing,
            concept_type_coverage=coverage,
            needs_review=needs_review,
            errors=errors,
        )
    
    def format_errors(self, result: ConceptSafetyResult) -> str:
        """Format concept safety errors for repair prompt."""
        if not result.errors:
            return ""
        
        lines = [
            f"Concept Safety Check: {result.recall:.1%} recall (threshold: {self.min_recall_threshold:.0%})",
            "",
            "Missing critical medical concepts:",
        ]
        
        for err in result.errors[:10]:
            lines.append(f"  - {err.message}")
        
        if len(result.missing_concepts) > 10:
            lines.append(f"  ... and {len(result.missing_concepts) - 10} more")
        
        lines.append("")
        lines.append("You MUST include these concepts in your simplified text.")
        
        return "\n".join(lines)
