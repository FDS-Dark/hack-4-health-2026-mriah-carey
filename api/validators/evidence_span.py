"""V2: Evidence-span validation for claim+evidence format."""

import logging
import re

from models.validation import (
    ErrorCode,
    Severity,
    ValidationError,
    ValidationResult,
)
from validators.base import BaseValidator

logger = logging.getLogger("validators.evidence_span")


class EvidenceSpanValidator(BaseValidator):
    """
    Validates that evidence spans are exact substrings of the original.
    
    This validator is useful when the simplified text uses a "Claim + Evidence" format
    where claims are supported by quoted evidence from the original.
    """
    
    name = "evidence_span_validator"
    
    EVIDENCE_PATTERNS = [
        re.compile(r'\((?:Source|Evidence|Quote|Original):\s*["\'](.+?)["\']\)', re.IGNORECASE | re.DOTALL),
        re.compile(r'\[(?:Source|Evidence|Quote):\s*["\'](.+?)["\']\]', re.IGNORECASE | re.DOTALL),
        re.compile(r'^>\s*(.+?)$', re.MULTILINE),
    ]
    
    CLAIM_INDICATORS = [
        re.compile(r'\bthe (?:report|scan|test|results?) (?:shows?|indicates?|reveals?|found)\b', re.IGNORECASE),
        re.compile(r'\baccording to the (?:report|scan|results?)\b', re.IGNORECASE),
        re.compile(r'\bdiagnosed with\b', re.IGNORECASE),
        re.compile(r'\bmeasures?\s+\d', re.IGNORECASE),
    ]
    
    def __init__(self, require_evidence_format: bool = False):
        self.require_evidence_format = require_evidence_format
    
    def _extract_evidence_spans(self, text: str) -> list[str]:
        """Extract all evidence spans from simplified text."""
        evidence_spans = []
        
        for pattern in self.EVIDENCE_PATTERNS:
            for match in pattern.finditer(text):
                evidence = match.group(1).strip()
                if evidence:
                    evidence_spans.append(evidence)
        
        return evidence_spans
    
    def _normalize_for_comparison(self, text: str) -> str:
        """Normalize text for substring comparison."""
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text.lower()
    
    def _is_exact_substring(self, evidence: str, original: str) -> bool:
        """Check if evidence is an exact substring of original."""
        norm_evidence = self._normalize_for_comparison(evidence)
        norm_original = self._normalize_for_comparison(original)
        return norm_evidence in norm_original
    
    def _find_ungrounded_claims(self, simplified: str, evidence_spans: list[str]) -> list[str]:
        """Find claims that should have evidence but don't."""
        ungrounded = []
        sentences = re.split(r'[.!?]\s+', simplified)
        
        for sentence in sentences:
            for pattern in self.CLAIM_INDICATORS:
                if pattern.search(sentence):
                    has_evidence = any(evidence in sentence or sentence in evidence for evidence in evidence_spans)
                    if not has_evidence:
                        ungrounded.append(sentence.strip())
                    break
        
        return ungrounded
    
    def validate(self, original_text: str, simplified_text: str) -> ValidationResult:
        """Validate evidence spans are exact substrings."""
        logger.info(f"{'='*60}")
        logger.info(f"EVIDENCE SPAN VALIDATOR - Starting validation")
        logger.info(f"{'='*60}")
        
        errors: list[ValidationError] = []
        
        evidence_spans = self._extract_evidence_spans(simplified_text)
        logger.debug(f"Found {len(evidence_spans)} evidence spans")
        
        for i, evidence in enumerate(evidence_spans):
            logger.debug(f"  Evidence {i+1}: '{evidence[:50]}...'")
            
            if not self._is_exact_substring(evidence, original_text):
                logger.warning(f"EVIDENCE NOT FOUND: '{evidence[:50]}...'")
                errors.append(ValidationError(
                    code=ErrorCode.EVIDENCE_NOT_FOUND,
                    severity=Severity.HIGH,
                    message="Evidence span is not an exact substring of the original",
                    summary_span=evidence[:100] + "..." if len(evidence) > 100 else evidence,
                    fix="Use an exact quote from the original text as evidence"
                ))
            else:
                logger.debug(f"  ✓ Evidence verified")
        
        if self.require_evidence_format:
            ungrounded = self._find_ungrounded_claims(simplified_text, evidence_spans)
            logger.debug(f"Found {len(ungrounded)} ungrounded claims")
            
            for claim in ungrounded:
                logger.warning(f"CLAIM WITHOUT EVIDENCE: '{claim[:50]}...'")
                errors.append(ValidationError(
                    code=ErrorCode.CLAIM_WITHOUT_EVIDENCE,
                    severity=Severity.MEDIUM,
                    message="Medical claim lacks supporting evidence from original",
                    summary_span=claim[:100] + "..." if len(claim) > 100 else claim,
                    fix="Add quoted evidence from the original to support this claim"
                ))
        
        if errors:
            logger.warning(f"EVIDENCE SPAN VALIDATOR - FAILED with {len(errors)} errors")
            return ValidationResult.failure(self.name, errors)
        
        logger.info("EVIDENCE SPAN VALIDATOR - PASSED ✓")
        return ValidationResult.success(self.name)
