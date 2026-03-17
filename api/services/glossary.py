"""Service for extracting and defining medical terms using UMLS concepts."""

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import medspacy

from clients.gemini import GeminiClient, get_gemini_client
from utils.file_handler import safe_json_loads

logger = logging.getLogger("services.glossary")


class GlossaryService:
    """Service to extract medical terms and provide definitions."""
    
    def __init__(
        self, 
        gemini_client: GeminiClient | None = None,
        max_workers: int = 8,
        max_terms: int = 30,
    ):
        """
        Initialize the glossary service.
        
        Args:
            gemini_client: Optional Gemini client. Uses default if not provided.
            max_workers: Max concurrent Gemini requests for definitions.
            max_terms: Maximum number of terms to include in glossary.
        """
        self.nlp = medspacy.load()
        self.client = gemini_client or get_gemini_client()
        self.max_workers = max_workers
        self.max_terms = max_terms
    
    def extract_medical_terms(self, text: str) -> list[str]:
        """Extract medical terms from text using medspacy."""
        doc = self.nlp(text)
        terms = set()
        
        # Extract named entities
        for ent in doc.ents:
            if ent.label_ in ['PROBLEM', 'TEST', 'TREATMENT', 'ANATOMY']:
                term = ent.text.strip()
                if len(term) > 2:
                    terms.add(term)
        
        # Also extract potential medical terms using patterns
        medical_patterns = [
            r'\b[A-Z]{2,}\b',  # Abbreviations like MRI, CT, ECG
            r'\b\d+\s*(mg|ml|mcg|units?|IU|mEq)\b',  # Dosages
        ]
        
        for pattern in medical_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    terms.add(' '.join(match))
                else:
                    terms.add(match)
        
        return sorted(list(terms))
    
    def get_term_definition(self, term: str) -> dict:
        """Get definition for a medical term using LLM."""
        prompt = f"""
Provide a simple, patient-friendly definition for this medical term.
Keep it brief (1-2 sentences) and use plain language (8th grade reading level).

Medical Term: {term}

Return JSON format:
{{
    "term": "{term}",
    "definition": "Simple definition here",
    "category": "diagnosis|test|treatment|anatomy|other"
}}
"""
        try:
            response = self.client.generate_json(prompt)
            definition_data = safe_json_loads(response)
            return {
                "term": term,
                "definition": definition_data.get("definition", "Definition not available"),
                "category": definition_data.get("category", "other")
            }
        except Exception as e:
            logger.warning(f"Failed to get definition for '{term}': {e}")
            return {
                "term": term,
                "definition": f"Medical term related to patient care.",
                "category": "other"
            }
    
    def get_definitions_batch(self, terms: list[str]) -> list[dict]:
        """
        Get definitions for multiple terms in parallel.
        
        Args:
            terms: List of medical terms to define.
            
        Returns:
            List of definition dictionaries in the same order as input terms.
        """
        if not terms:
            return []
        
        logger.info(f"Fetching definitions for {len(terms)} terms in parallel (max {self.max_workers} workers)")
        
        # Pre-allocate results to maintain order
        results = [None] * len(terms)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(self.get_term_definition, term): idx
                for idx, term in enumerate(terms)
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                term = terms[idx]
                try:
                    result = future.result()
                    results[idx] = result
                    completed += 1
                    logger.debug(f"  [{completed}/{len(terms)}] Got definition for '{term}'")
                except Exception as e:
                    logger.error(f"  Failed to get definition for '{term}': {e}")
                    results[idx] = {
                        "term": term,
                        "definition": "Medical term found in your report.",
                        "category": "other"
                    }
        
        logger.info(f"Completed {completed}/{len(terms)} definitions")
        return results
    
    def create_glossary(self, original_text: str, simplified_text: str) -> list[dict]:
        """
        Create a glossary from both original and simplified text.
        
        Extracts medical terms and fetches definitions in parallel.
        """
        try:
            # Extract terms from both texts
            logger.info("Extracting medical terms...")
            original_terms = self.extract_medical_terms(original_text)
            simplified_terms = self.extract_medical_terms(simplified_text)
            
            # Combine and deduplicate
            all_terms = sorted(set(original_terms + simplified_terms))
            logger.info(f"Found {len(all_terms)} unique terms")
            
            # Limit to max terms for performance
            if len(all_terms) > self.max_terms:
                logger.info(f"Limiting to {self.max_terms} terms")
                all_terms = all_terms[:self.max_terms]
                
        except Exception as e:
            logger.error(f"Error extracting medical terms: {e}")
            return []
        
        # Get definitions in parallel
        glossary = self.get_definitions_batch(all_terms)
        
        return glossary
