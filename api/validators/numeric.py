"""V1: Numeric and unit preservation validator."""

import logging
import re
from dataclasses import dataclass

from models.validation import (
    ErrorCode,
    Severity,
    ValidationError,
    ValidationResult,
)
from validators.base import BaseValidator
from utils.nlp import extract_numbers_with_context, ExtractedNumber

logger = logging.getLogger("validators.numeric")


class NumericValidator(BaseValidator):
    """
    Validates numeric values, units, and laterality preservation.
    
    Uses NLP-based number extraction from utils.nlp module.
    Compares sets of extracted numbers between original and simplified.
    
    Focuses on MEDICALLY RELEVANT numbers:
    - Measurements with units (4 mm, 10 cm, 48.2 kg, etc.)
    - Vitals (BP, pulse, BMI)
    - Lab values
    - Dosages
    """
    
    name = "numeric_validator"
    
    # Pattern for BP-style readings (e.g., 122/84)
    BP_PATTERN = re.compile(r'(\d{2,3})\s*/\s*(\d{2,3})(?:\s*mmHg)?', re.IGNORECASE)
    
    # Patterns for dates (to exclude)
    DATE_PATTERNS = [
        re.compile(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b'),  # 6/6/2023
        re.compile(r'\b\d{1,2}-\d{1,2}-\d{2,4}\b'),  # 6-6-2023
        re.compile(r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}', re.IGNORECASE),
        re.compile(r'\b\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*', re.IGNORECASE),
    ]
    
    # Patterns for numbers to always ignore
    IGNORE_PATTERNS = [
        re.compile(r'\b(?:19|20)\d{2}\b'),  # Years 1900-2099
        re.compile(r'\bpage\s*\d+', re.IGNORECASE),
        re.compile(r'\b\d+\s*(?:of|/)\s*\d+\s*$', re.MULTILINE),  # "page 2 of 3"
        re.compile(r'#\d+'),
        re.compile(r'\d{3}[-.]?\d{3}[-.]?\d{4}'),  # Phone numbers
        re.compile(r'\b\d{7,}\b'),  # Long ID numbers (7+ digits)
        re.compile(r'\bMRN[:\s]*\d+', re.IGNORECASE),  # MRN numbers
        re.compile(r'\d{1,2}:\d{2}(?:\s*(?:AM|PM))?', re.IGNORECASE),  # Times
    ]
    
    # Laterality terms
    LATERALITY_LEFT = {"left", "left-sided", "l."}
    LATERALITY_RIGHT = {"right", "right-sided", "r."}
    
    def __init__(self, min_significant_value: float = 3.0):
        """
        Initialize the numeric validator.
        
        Args:
            min_significant_value: Minimum value to flag as missing.
                Values <= this might just be list numbers, not measurements.
        """
        self.min_significant_value = min_significant_value
    
    def _is_in_ignored_context(self, text: str, position: int) -> bool:
        """Check if a position in text is part of an ignored pattern."""
        # Check date patterns
        for pattern in self.DATE_PATTERNS:
            for m in pattern.finditer(text):
                if m.start() <= position < m.end():
                    return True
        
        # Check ignore patterns
        for pattern in self.IGNORE_PATTERNS:
            for m in pattern.finditer(text):
                if m.start() <= position < m.end():
                    return True
        
        return False
    
    def _extract_bp_values(self, text: str) -> list[ExtractedNumber]:
        """Extract blood pressure values (special case for X/Y format)."""
        results = []
        text_lower = text.lower()
        
        for match in self.BP_PATTERN.finditer(text_lower):
            if self._is_in_ignored_context(text_lower, match.start()):
                continue
            
            systolic = match.group(1)
            diastolic = match.group(2)
            
            start = max(0, match.start() - 20)
            end = min(len(text), match.end() + 20)
            context = text[start:end]
            
            results.append(ExtractedNumber(
                value=float(systolic),
                text=systolic,
                unit="mmhg",
                context=context
            ))
            results.append(ExtractedNumber(
                value=float(diastolic),
                text=diastolic,
                unit="mmhg",
                context=context
            ))
        
        return results
    
    def _extract_measurements(self, text: str) -> list[ExtractedNumber]:
        """Extract all medical measurements from text using NLP utilities."""
        # Use NLP-based extraction
        measurements = extract_numbers_with_context(text)
        
        # Filter out ignored patterns
        filtered = []
        for m in measurements:
            # Skip if in ignored context
            # We need to find the position in the original text
            # For now, check if the context contains ignored patterns
            if any(p.search(m.context) for p in self.IGNORE_PATTERNS):
                continue
            if any(p.search(m.context) for p in self.DATE_PATTERNS):
                continue
            
            filtered.append(m)
        
        # Add BP values (special handling)
        bp_values = self._extract_bp_values(text)
        filtered.extend(bp_values)
        
        return filtered
    
    def _normalize_unit(self, unit: str | None) -> str | None:
        """Normalize a unit for comparison."""
        if not unit:
            return None
        unit = unit.lower().strip()
        unit_map = {
            "ml": "ml", "ml": "ml", "l": "l", 
            "cm3": "cc", "cm³": "cc", "cc": "cc",
            "microgram": "mcg", "μg": "mcg", "ug": "mcg",
            "inches": "in", '"': "in",
            "feet": "ft", "'": "ft",
        }
        return unit_map.get(unit, unit)
    
    def _extract_laterality(self, text: str) -> tuple[set[str], set[str]]:
        """Extract laterality mentions (left, right) from text."""
        text_lower = text.lower()
        words = set(re.findall(r'\b\w+(?:-\w+)?\.?\b', text_lower))
        
        left_found = words & self.LATERALITY_LEFT
        right_found = words & self.LATERALITY_RIGHT
        
        return left_found, right_found
    
    def validate(self, original_text: str, simplified_text: str) -> ValidationResult:
        """Validate numeric/unit/laterality preservation using SET COMPARISON."""
        logger.info(f"{'='*60}")
        logger.info(f"NUMERIC VALIDATOR - Starting validation")
        logger.info(f"{'='*60}")
        
        errors: list[ValidationError] = []
        
        # Extract measurements from BOTH texts using NLP
        orig_measurements = self._extract_measurements(original_text)
        simp_measurements = self._extract_measurements(simplified_text)
        
        logger.debug(f"Original measurements ({len(orig_measurements)}):")
        for m in orig_measurements[:10]:
            logger.debug(f"  {m.value} {m.unit or ''}")
        
        logger.debug(f"Simplified measurements ({len(simp_measurements)}):")
        for m in simp_measurements[:10]:
            logger.debug(f"  {m.value} {m.unit or ''}")
        
        # Create sets for comparison (round to handle floating point)
        orig_values = {round(m.value, 2) for m in orig_measurements}
        simp_values = {round(m.value, 2) for m in simp_measurements}
        
        # Check for missing measurements
        missing_values = orig_values - simp_values
        
        for missing in missing_values:
            # Find the original measurement for context
            orig_m = next((m for m in orig_measurements if round(m.value, 2) == missing), None)
            if orig_m:
                # Only flag significant values (not list numbers like 1, 2, 3)
                # Also require a unit for small values
                is_significant = (
                    missing > self.min_significant_value or 
                    orig_m.unit is not None
                )
                
                if is_significant:
                    logger.warning(f"MISSING: {missing} {orig_m.unit or ''}")
                    errors.append(ValidationError(
                        code=ErrorCode.NUMERIC_MISSING,
                        severity=Severity.HIGH,
                        message=f"Medical measurement '{orig_m.text} {orig_m.unit or ''}' is missing from simplified text",
                        source_evidence=orig_m.context[:100],
                        fix=f"Include the measurement '{orig_m.text} {orig_m.unit or ''}' in the simplified text"
                    ))
        
        # Check for unit mismatches (same value, different unit)
        for orig_m in orig_measurements:
            if not orig_m.unit:
                continue
            for simp_m in simp_measurements:
                if not simp_m.unit:
                    continue
                if round(orig_m.value, 2) == round(simp_m.value, 2):
                    orig_unit = self._normalize_unit(orig_m.unit)
                    simp_unit = self._normalize_unit(simp_m.unit)
                    
                    if orig_unit != simp_unit:
                        # Check if it's a valid conversion
                        valid_conversions = {
                            ("cm", "mm"), ("mm", "cm"),
                            ("kg", "lb"), ("lb", "kg"),
                            ("in", "cm"), ("cm", "in"),
                        }
                        if (orig_unit, simp_unit) not in valid_conversions:
                            logger.error(f"UNIT MISMATCH: {orig_m.value} {orig_unit} → {simp_unit}")
                            errors.append(ValidationError(
                                code=ErrorCode.UNIT_MISMATCH,
                                severity=Severity.CRITICAL,
                                message=f"Unit changed from '{orig_unit}' to '{simp_unit}' for value {orig_m.value}",
                                source_evidence=f"{orig_m.text} {orig_m.unit}",
                                summary_span=f"{simp_m.text} {simp_m.unit}",
                                fix=f"Use the original unit '{orig_unit}' instead of '{simp_unit}'"
                            ))
        
        # Check laterality preservation
        orig_left, orig_right = self._extract_laterality(original_text)
        simp_left, simp_right = self._extract_laterality(simplified_text)
        
        logger.debug(f"Laterality - Orig: L={orig_left}, R={orig_right} | Simp: L={simp_left}, R={simp_right}")
        
        # Only flag clear flips
        if orig_left and not orig_right and simp_right and not simp_left:
            logger.error("LATERALITY FLIP: LEFT → RIGHT")
            errors.append(ValidationError(
                code=ErrorCode.LATERALITY_FLIP,
                severity=Severity.CRITICAL,
                message="Laterality flipped: original says LEFT but simplified says RIGHT",
                source_evidence="left",
                summary_span="right",
                fix="Change 'right' to 'left' to match original"
            ))
        
        if orig_right and not orig_left and simp_left and not simp_right:
            logger.error("LATERALITY FLIP: RIGHT → LEFT")
            errors.append(ValidationError(
                code=ErrorCode.LATERALITY_FLIP,
                severity=Severity.CRITICAL,
                message="Laterality flipped: original says RIGHT but simplified says LEFT",
                source_evidence="right",
                summary_span="left",
                fix="Change 'left' to 'right' to match original"
            ))
        
        if errors:
            logger.warning(f"NUMERIC VALIDATOR - FAILED with {len(errors)} errors")
            return ValidationResult.failure(self.name, errors)
        
        logger.info("NUMERIC VALIDATOR - PASSED ✓")
        return ValidationResult.success(self.name)
