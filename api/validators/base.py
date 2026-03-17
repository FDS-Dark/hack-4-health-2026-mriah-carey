"""Base validator interface."""

import logging
from abc import ABC, abstractmethod

from models.validation import ValidationResult

# Configure logging for validators
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger("validators")


class BaseValidator(ABC):
    """Abstract base class for all validators."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this validator."""
        ...
    
    @property
    def is_hard_gate(self) -> bool:
        """Whether failure of this validator should block delivery."""
        return True
    
    @abstractmethod
    def validate(self, original_text: str, simplified_text: str) -> ValidationResult:
        """
        Validate the simplified text against the original.
        
        Args:
            original_text: The original medical report text.
            simplified_text: The simplified version to validate.
        
        Returns:
            ValidationResult with pass/fail status and any errors.
        """
        ...
