"""Abstract base classes for text parsing and report extraction."""

from abc import ABC, abstractmethod
from models.report_schema import ParsedReport


class TextParser(ABC):
    """Abstract base class for text preprocessing and parsing."""
    
    @abstractmethod
    def parse(self, text: str) -> str:
        """
        Preprocess/parse text.
        
        Args:
            text: Raw text input
            
        Returns:
            Cleaned/processed text
        """
        pass


class ReportExtractor(ABC):
    """Abstract base class for extracting structured data from medical reports."""
    
    @abstractmethod
    def extract(self, text: str) -> ParsedReport:
        """
        Extract main ideas and supporting details from text.
        
        Args:
            text: Medical report text (can be preprocessed)
            
        Returns:
            ParsedReport with main ideas and details
        """
        pass
