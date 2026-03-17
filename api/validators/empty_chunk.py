"""Empty chunk and meta-talk validator."""

import logging
import re

from models.validation import (
    ErrorCode,
    Severity,
    ValidationError,
    ValidationResult,
)
from validators.base import BaseValidator

logger = logging.getLogger("validators.empty_chunk")


class EmptyChunkValidator(BaseValidator):
    """
    Validates that chunks are not empty and don't contain meta-talk.
    
    Flags:
    - Empty or near-empty original text
    - AI assistant meta-talk responses ("I need the text...", etc.)
    - Responses that don't contain actual medical content
    """
    
    name = "empty_chunk_validator"
    
    # Minimum content length for a valid chunk
    MIN_ORIGINAL_LENGTH = 20
    MIN_SIMPLIFIED_LENGTH = 50
    
    # Patterns that indicate AI meta-talk (not actual content)
    META_TALK_PATTERNS = [
        re.compile(r"I need the.*text", re.IGNORECASE),
        re.compile(r"please provide.*text", re.IGNORECASE),
        re.compile(r"I cannot.*without", re.IGNORECASE),
        re.compile(r"I don't have.*information", re.IGNORECASE),
        re.compile(r"please share.*report", re.IGNORECASE),
        re.compile(r"provide.*medical.*text", re.IGNORECASE),
        re.compile(r"I'm unable to", re.IGNORECASE),
        re.compile(r"I can't.*without", re.IGNORECASE),
        re.compile(r"no (?:medical )?text (?:was )?provided", re.IGNORECASE),
        re.compile(r"waiting for.*input", re.IGNORECASE),
        re.compile(r"send me the.*text", re.IGNORECASE),
    ]
    
    def __init__(
        self,
        min_original_length: int = 20,
        min_simplified_length: int = 50,
    ):
        self.min_original_length = min_original_length
        self.min_simplified_length = min_simplified_length
    
    def _is_meta_talk(self, text: str) -> bool:
        """Check if text is AI meta-talk rather than content."""
        for pattern in self.META_TALK_PATTERNS:
            if pattern.search(text):
                return True
        return False
    
    def _get_content_density(self, text: str) -> float:
        """
        Calculate content density - ratio of alphanumeric chars to total length.
        
        Low density indicates mostly whitespace/formatting.
        """
        if not text:
            return 0.0
        
        alphanumeric = sum(1 for c in text if c.isalnum())
        return alphanumeric / len(text)
    
    def validate(self, original_text: str, simplified_text: str) -> ValidationResult:
        """Validate that chunk has real content."""
        logger.info(f"{'='*60}")
        logger.info(f"EMPTY CHUNK VALIDATOR - Starting validation")
        logger.info(f"{'='*60}")
        
        errors: list[ValidationError] = []
        
        original_stripped = original_text.strip()
        simplified_stripped = simplified_text.strip()
        
        logger.debug(f"Original length: {len(original_stripped)}, Simplified length: {len(simplified_stripped)}")
        
        # Check if original is empty
        if len(original_stripped) < self.min_original_length:
            logger.warning(f"EMPTY ORIGINAL: Only {len(original_stripped)} chars")
            errors.append(ValidationError(
                code=ErrorCode.EMPTY_CONTENT,
                severity=Severity.MEDIUM,
                message=f"Original text is too short ({len(original_stripped)} chars)",
                source_evidence=original_stripped[:50] if original_stripped else "(empty)",
                fix="Skip this chunk - no content to simplify"
            ))
        
        # Check if simplified is empty
        if len(simplified_stripped) < self.min_simplified_length:
            logger.warning(f"EMPTY SIMPLIFIED: Only {len(simplified_stripped)} chars")
            errors.append(ValidationError(
                code=ErrorCode.EMPTY_CONTENT,
                severity=Severity.MEDIUM,
                message=f"Simplified text is too short ({len(simplified_stripped)} chars)",
                summary_span=simplified_stripped[:50] if simplified_stripped else "(empty)",
                fix="Regenerate simplified content or skip this chunk"
            ))
        
        # Check for AI meta-talk in simplified text
        if self._is_meta_talk(simplified_stripped):
            logger.error(f"META-TALK DETECTED: '{simplified_stripped[:60]}...'")
            errors.append(ValidationError(
                code=ErrorCode.META_TALK_DETECTED,
                severity=Severity.HIGH,
                message="Simplified text contains AI meta-talk instead of actual content",
                summary_span=simplified_stripped[:100],
                fix="Regenerate with actual medical content, or skip if original is empty"
            ))
        
        # Check content density (very low = mostly whitespace/formatting)
        orig_density = self._get_content_density(original_stripped)
        simp_density = self._get_content_density(simplified_stripped)
        
        logger.debug(f"Content density - Original: {orig_density:.2f}, Simplified: {simp_density:.2f}")
        
        if orig_density < 0.3 and len(original_stripped) > 10:
            logger.warning(f"LOW DENSITY ORIGINAL: {orig_density:.2%}")
            errors.append(ValidationError(
                code=ErrorCode.EMPTY_CONTENT,
                severity=Severity.LOW,
                message=f"Original text has low content density ({orig_density:.0%})",
                source_evidence=original_stripped[:50],
                fix="May be formatting-heavy chunk - consider skipping"
            ))
        
        if errors:
            # For empty chunk errors, we use MEDIUM severity overall
            # because these aren't "dangerous" just "useless"
            logger.warning(f"EMPTY CHUNK VALIDATOR - FAILED with {len(errors)} errors")
            return ValidationResult.failure(self.name, errors)
        
        logger.info("EMPTY CHUNK VALIDATOR - PASSED ✓")
        return ValidationResult.success(self.name)
