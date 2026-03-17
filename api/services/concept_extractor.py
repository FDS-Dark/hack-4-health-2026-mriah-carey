"""Extract medical concepts from text for concept preservation during simplification."""

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger("services.concept_extractor")


@dataclass
class MedicalConcept:
    """A medical concept extracted from text."""
    text: str
    cui: str | None = None
    semantic_type: str | None = None
    importance: str = "normal"  # "critical", "high", "normal"


# Critical concept patterns that MUST be preserved
CRITICAL_PATTERNS = [
    # Diagnoses and conditions
    r'\b(?:diagnosed with|diagnosis of|positive for|confirmed)\s+[\w\s]+',
    # Medications with dosages
    r'\b\d+\s*(?:mg|mcg|g|ml|units?)\b',
    # Lab values
    r'\b\d+\.?\d*\s*(?:mg/dL|g/dL|mmol/L|mEq/L|U/L|IU/L|ng/mL|pg/mL)\b',
    # Vital signs
    r'\b(?:blood pressure|BP|heart rate|HR|temperature|temp|pulse|SpO2|oxygen)\s*:?\s*\d+',
    # Anatomical locations with laterality
    r'\b(?:left|right|bilateral)\s+[\w\s]+',
    # Measurements
    r'\b\d+\.?\d*\s*(?:mm|cm|m|inches?|feet|kg|lbs?|pounds?)\b',
    # Spinal levels
    r'\b[CTLS]\d+(?:-[CTLS]?\d+)?\b',
    # Percentages
    r'\b\d+\.?\d*\s*%\b',
    # Dates and durations
    r'\b\d+\s*(?:days?|weeks?|months?|years?)\b',
]

# High importance semantic types
HIGH_IMPORTANCE_TYPES = {
    "T047",  # Disease or Syndrome
    "T048",  # Mental or Behavioral Dysfunction  
    "T184",  # Sign or Symptom
    "T121",  # Pharmacologic Substance
    "T200",  # Clinical Drug
    "T060",  # Diagnostic Procedure
    "T061",  # Therapeutic or Preventive Procedure
    "T033",  # Finding
    "T034",  # Laboratory or Test Result
    "T191",  # Neoplastic Process
    "T046",  # Pathologic Function
}


class ConceptExtractor:
    """
    Extracts medical concepts from text that must be preserved during simplification.
    
    Uses pattern matching and optionally UMLS lookups.
    """
    
    def __init__(self, opensearch_client=None):
        self._opensearch_client = opensearch_client
        self._umls_index = None
        self._nlp = None
    
    def _get_umls_index(self):
        """Lazy load UMLS index."""
        if self._umls_index is not None:
            return self._umls_index
        
        try:
            from services.umls_service import UMLSSynonymIndex
            from clients.opensearch import get_opensearch_client
            
            client = self._opensearch_client or get_opensearch_client()
            if client.connect():
                self._umls_index = UMLSSynonymIndex(client)
                logger.info("UMLS index loaded for concept extraction")
        except Exception as e:
            logger.debug(f"UMLS not available: {e}")
        
        return self._umls_index
    
    def _get_nlp(self):
        """Lazy load spaCy."""
        if self._nlp is None:
            try:
                import spacy
                self._nlp = spacy.load("en_core_web_sm")
            except Exception as e:
                logger.debug(f"spaCy not available: {e}")
        return self._nlp
    
    def extract_critical_patterns(self, text: str) -> list[str]:
        """Extract text matching critical patterns."""
        matches = []
        for pattern in CRITICAL_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                matches.append(match.group(0).strip())
        return list(set(matches))
    
    def extract_concepts(self, text: str) -> list[MedicalConcept]:
        """
        Extract medical concepts from text.
        
        Returns list of concepts sorted by importance.
        """
        concepts = []
        seen = set()
        
        # 1. Extract critical patterns first
        critical_matches = self.extract_critical_patterns(text)
        for match in critical_matches:
            if match.lower() not in seen:
                concepts.append(MedicalConcept(
                    text=match,
                    importance="critical",
                ))
                seen.add(match.lower())
        
        # 2. Extract UMLS concepts
        umls_index = self._get_umls_index()
        nlp = self._get_nlp()
        
        if umls_index and nlp:
            doc = nlp(text)
            
            # Check noun chunks
            for chunk in doc.noun_chunks:
                if chunk.text.lower() in seen:
                    continue
                    
                matches = umls_index.search_term(chunk.text, limit=1)
                if matches and matches[0].score > 5.0:
                    match = matches[0]
                    importance = "high" if any(
                        t in HIGH_IMPORTANCE_TYPES 
                        for t in match.semantic_types
                    ) else "normal"
                    
                    concepts.append(MedicalConcept(
                        text=chunk.text,
                        cui=match.cui,
                        semantic_type=match.semantic_types[0] if match.semantic_types else None,
                        importance=importance,
                    ))
                    seen.add(chunk.text.lower())
            
            # Check named entities
            for ent in doc.ents:
                if ent.text.lower() in seen:
                    continue
                if ent.label_ in {"ORG", "GPE", "PERSON"}:
                    continue
                    
                matches = umls_index.search_term(ent.text, limit=1)
                if matches and matches[0].score > 5.0:
                    match = matches[0]
                    importance = "high" if any(
                        t in HIGH_IMPORTANCE_TYPES
                        for t in match.semantic_types
                    ) else "normal"
                    
                    concepts.append(MedicalConcept(
                        text=ent.text,
                        cui=match.cui,
                        semantic_type=match.semantic_types[0] if match.semantic_types else None,
                        importance=importance,
                    ))
                    seen.add(ent.text.lower())
        
        # Sort by importance
        importance_order = {"critical": 0, "high": 1, "normal": 2}
        concepts.sort(key=lambda c: importance_order.get(c.importance, 2))
        
        return concepts
    
    def format_for_prompt(self, concepts: list[MedicalConcept], max_concepts: int = 30) -> str:
        """
        Format extracted concepts for inclusion in LLM prompt.
        
        Returns a string that can be injected into the simplifier prompt.
        """
        if not concepts:
            return ""
        
        # Prioritize critical and high importance
        critical = [c for c in concepts if c.importance == "critical"][:15]
        high = [c for c in concepts if c.importance == "high"][:10]
        normal = [c for c in concepts if c.importance == "normal"][:5]
        
        selected = critical + high + normal
        
        lines = [
            "CRITICAL - You MUST preserve these medical concepts in your simplified output:",
            "(You should explain them in simpler terms, but do NOT omit them)",
            ""
        ]
        
        for c in selected:
            if c.importance == "critical":
                lines.append(f"  * {c.text} [MUST INCLUDE]")
            elif c.importance == "high":
                lines.append(f"  * {c.text}")
            else:
                lines.append(f"  - {c.text}")
        
        return "\n".join(lines)


# Singleton
_extractor: ConceptExtractor | None = None


def get_concept_extractor(opensearch_client=None) -> ConceptExtractor:
    """Get or create the concept extractor singleton."""
    global _extractor
    if _extractor is None:
        _extractor = ConceptExtractor(opensearch_client=opensearch_client)
    return _extractor


def extract_concepts_for_prompt(text: str, opensearch_client=None) -> str:
    """
    Convenience function to extract concepts and format for prompt.
    
    Returns formatted string ready to inject into simplifier prompt.
    """
    extractor = get_concept_extractor(opensearch_client)
    concepts = extractor.extract_concepts(text)
    return extractor.format_for_prompt(concepts)
