"""Service for comparing original and simplified medical text."""

import logging
import re

import medspacy
from medspacy.ner import TargetRule

logger = logging.getLogger("services.comparison")


class ComparisonService:
    """Service for comparing entities, numbers, and negations between texts."""
    
    def __init__(self):
        # Suppress medspacy warnings during load
        logging.getLogger("medspacy").setLevel(logging.WARNING)
        self.nlp = medspacy.load()
    
    def _sanitize_keyword(self, keyword: str) -> str:
        """
        Sanitize a keyword for use with medspacy TargetRule.
        
        Removes or escapes characters that medspacy doesn't handle well.
        """
        # Remove leading/trailing whitespace
        keyword = keyword.strip()
        
        # Remove periods, brackets, and other problematic characters
        # These can cause regex issues in medspacy
        keyword = re.sub(r'[.\[\](){}^$*+?|\\]', '', keyword)
        
        # Collapse multiple spaces
        keyword = re.sub(r'\s+', ' ', keyword)
        
        return keyword.strip()
    
    def _extract_entities(self, text: str, keywords: list[str]) -> list[str]:
        """Extract entities from text based on keywords."""
        try:
            matcher = self.nlp.get_pipe("medspacy_target_matcher")
            
            # Add rules for each sanitized keyword
            for term in keywords:
                sanitized = self._sanitize_keyword(term)
                if sanitized and len(sanitized) >= 2:  # Skip empty or very short terms
                    try:
                        rule = TargetRule(sanitized, "MEDICAL_CONCEPT")
                        matcher.add(rule)
                    except Exception as e:
                        logger.debug(f"Could not add rule for '{sanitized}': {e}")
            
            doc = self.nlp(text)
            entities = [ent.text.lower() for ent in doc.ents]
            return entities
            
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
            return []
    
    def _extract_numbers(self, text: str) -> set[str]:
        """Extract numbers from text."""
        return set(re.findall(r'\d+[\d\-%]*', text))
    
    def _extract_negations(self, text: str) -> set[str]:
        """Extract negation words from text."""
        return set(re.findall(r'\b(no|not|without|none|never|did not|does not)\b', text.lower()))
    
    def _validate_entities(self, expected_terms: set[str], found_entities: set[str]) -> tuple[set[str], set[str]]:
        """Check if found_entities contains all expected_terms."""
        expected_set = set(expected_terms)
        found_set = set(found_entities)
        missing = expected_set - found_set
        extra = found_set - expected_set
        return missing, extra
    
    def _validate_numbers(self, orig: set[str], simp: set[str]) -> tuple[set[str], set[str]]:
        """Compare numbers between original and simplified."""
        missing = orig - simp
        extra = simp - orig
        return missing, extra
    
    def validate_negation(self, orig: set[str], simp: set[str]) -> set[str]:
        """Find negations in original that are missing from simplified."""
        return orig - simp
    
    def check_entries(self, simplified_text: str, keywords: list[str]) -> tuple[set[str], set[str]]:
        """Check for missing/extra entities in simplified text."""
        keywords_lower = [kw.lower() for kw in keywords]
        simplified_entities = self._extract_entities(simplified_text, keywords_lower)
        entity_missing, entity_extra = self._validate_entities(set(keywords_lower), set(simplified_entities))
        return entity_missing, entity_extra
    
    def check_numbers(self, original_text: str, simplified_text: str) -> tuple[set[str], set[str]]:
        """Check for missing/extra numbers in simplified text."""
        original_numbers = self._extract_numbers(original_text)
        simplified_numbers = self._extract_numbers(simplified_text)
        number_missing, number_extra = self._validate_numbers(original_numbers, simplified_numbers)
        return number_missing, number_extra
    
    def check_negations(self, original_text: str, simplified_text: str) -> set[str]:
        """Check for missing negations in simplified text."""
        original_negations = self._extract_negations(original_text)
        simplified_negations = self._extract_negations(simplified_text)
        negation_missing = self.validate_negation(original_negations, simplified_negations)
        return negation_missing
