"""
UMLS OpenSearch Service for medical term normalization and synonym lookup.

Uses the OpenSearchClient for UMLS lookups and provides synonym/canonical term
functionality for medical text normalization.
"""

import logging
import re
from dataclasses import dataclass
from functools import lru_cache

from clients.opensearch import OpenSearchClient, get_opensearch_client

logger = logging.getLogger(__name__)


@dataclass
class UMLSMatch:
    """A matched UMLS concept."""
    cui: str
    term: str
    preferred_term: str
    score: float
    semantic_types: list[str]
    is_preferred: bool = False


class UMLSSynonymIndex:
    """
    OpenSearch-backed UMLS synonym index.
    
    Provides synonym lookup and canonical term resolution using UMLS concepts.
    """
    
    def __init__(self, client: OpenSearchClient | None = None):
        """
        Initialize the synonym index.
        
        Args:
            client: OpenSearch client. Uses default if not provided.
        """
        self.client = client or get_opensearch_client()
        self._cache_canonical: dict[str, str | None] = {}
        self._cache_synonyms: dict[str, set[str]] = {}
    
    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        if not text:
            return ""
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def search_term(self, term: str, limit: int = 10) -> list[UMLSMatch]:
        """
        Search for UMLS concepts matching a term.
        
        Returns list of matches sorted by relevance score.
        """
        hits = self.client.search_term(term, limit=limit)
        
        matches = []
        for hit in hits:
            src = hit.get("_source", {})
            matches.append(UMLSMatch(
                cui=src.get("cui", ""),
                term=src.get("term", ""),
                preferred_term=src.get("term", ""),
                score=hit.get("_score", 0.0),
                semantic_types=src.get("semantic_types", []),
                is_preferred=src.get("is_preferred", False),
            ))
        
        return matches
    
    def get_concept(self, cui: str) -> dict | None:
        """Get full concept details by CUI."""
        return self.client.get_concept(cui)
    
    def get_canonical(self, term: str) -> str | None:
        """
        Get the canonical (preferred) term for a given term.
        
        Returns the UMLS preferred term if found, None otherwise.
        """
        normalized = self._normalize(term)
        
        if normalized in self._cache_canonical:
            return self._cache_canonical[normalized]
        
        matches = self.search_term(term, limit=5)
        
        if not matches:
            self._cache_canonical[normalized] = None
            return None
        
        best = matches[0]
        
        if best.score > 5:
            concept = self.get_concept(best.cui)
            if concept:
                preferred = concept.get("preferred_term", best.term)
                self._cache_canonical[normalized] = preferred
                return preferred
        
        self._cache_canonical[normalized] = None
        return None
    
    def get_synonyms(self, term: str) -> set[str]:
        """Get all synonyms for a term (terms sharing the same CUI)."""
        normalized = self._normalize(term)
        
        if normalized in self._cache_synonyms:
            return self._cache_synonyms[normalized]
        
        matches = self.search_term(term, limit=1)
        
        if not matches:
            self._cache_synonyms[normalized] = set()
            return set()
        
        cui = matches[0].cui
        concept = self.get_concept(cui)
        
        if concept:
            terms = set(self._normalize(t) for t in concept.get("terms", []))
            self._cache_synonyms[normalized] = terms
            return terms
        
        self._cache_synonyms[normalized] = set()
        return set()
    
    def are_synonyms(self, term1: str, term2: str) -> bool:
        """Check if two terms are synonyms (map to the same CUI)."""
        norm1 = self._normalize(term1)
        norm2 = self._normalize(term2)
        
        if norm1 == norm2:
            return True
        
        matches1 = self.search_term(term1, limit=1)
        matches2 = self.search_term(term2, limit=1)
        
        if not matches1 or not matches2:
            return False
        
        return matches1[0].cui == matches2[0].cui
    
    def canonicalize(self, text: str) -> str:
        """Replace known terms in text with their canonical forms."""
        if not text:
            return text
        
        words = text.split()
        result = []
        i = 0
        
        while i < len(words):
            matched = False
            
            for n in [3, 2, 1]:
                if i + n <= len(words):
                    phrase = ' '.join(words[i:i+n])
                    canonical = self.get_canonical(phrase)
                    if canonical:
                        result.append(canonical)
                        i += n
                        matched = True
                        break
            
            if not matched:
                result.append(words[i])
                i += 1
        
        return ' '.join(result)
    
    def get_semantic_types(self, term: str) -> list[str]:
        """Get semantic types (TUIs) for a term."""
        matches = self.search_term(term, limit=1)
        if matches:
            return matches[0].semantic_types
        return []


# Fallback synonyms for common clinical phrases
FALLBACK_SYNONYMS = [
    {"normal", "unremarkable", "within normal limits", "wnl", "negative", "benign", "no abnormality"},
    {"no", "without", "absent", "none", "no evidence of", "no sign of", "no signs of", "not seen", "not identified"},
    {"recommend", "recommends", "recommended", "recommendation", "advised", "advise", "suggest", "suggested"},
    {"bilateral", "bilaterally", "both", "both sides"},
    {"ct", "ct scan", "computed tomography", "cat scan"},
    {"doctor", "physician", "your doctor", "doctors"},
]


class HybridSynonymIndex:
    """
    Hybrid synonym index combining UMLS OpenSearch and fallback synonyms.
    
    First checks UMLS via OpenSearch, then falls back to hardcoded synonyms.
    """
    
    def __init__(
        self,
        umls_index: UMLSSynonymIndex | None = None,
        fallback_synonyms: list[set[str]] | None = None
    ):
        self.umls_index = umls_index
        self.fallback_synonyms = fallback_synonyms or FALLBACK_SYNONYMS
        
        self._fallback_lookup: dict[str, set[str]] = {}
        self._fallback_canonical: dict[str, str] = {}
        
        for group in self.fallback_synonyms:
            canonical = min(group, key=len)
            for term in group:
                normalized = term.lower().strip()
                self._fallback_lookup[normalized] = group
                self._fallback_canonical[normalized] = canonical
    
    def get_canonical(self, term: str) -> str | None:
        """Get canonical form, checking UMLS first then fallback."""
        normalized = term.lower().strip()
        
        if self.umls_index:
            canonical = self.umls_index.get_canonical(term)
            if canonical:
                return canonical
        
        if normalized in self._fallback_canonical:
            return self._fallback_canonical[normalized]
        
        return None
    
    def get_synonyms(self, term: str) -> set[str]:
        """Get all synonyms, checking UMLS first then fallback."""
        normalized = term.lower().strip()
        
        if self.umls_index:
            synonyms = self.umls_index.get_synonyms(term)
            if synonyms:
                return synonyms
        
        if normalized in self._fallback_lookup:
            return self._fallback_lookup[normalized]
        
        return set()
    
    def are_synonyms(self, term1: str, term2: str) -> bool:
        """Check if two terms are synonyms."""
        norm1 = term1.lower().strip()
        norm2 = term2.lower().strip()
        
        if norm1 == norm2:
            return True
        
        if self.umls_index and self.umls_index.are_synonyms(term1, term2):
            return True
        
        if norm1 in self._fallback_lookup:
            return norm2 in self._fallback_lookup[norm1]
        
        return False
    
    def canonicalize(self, text: str) -> str:
        """Canonicalize text using both UMLS and fallback synonyms."""
        if not text:
            return text
        
        if self.umls_index:
            text = self.umls_index.canonicalize(text)
        
        words = text.split()
        result = []
        i = 0
        
        while i < len(words):
            matched = False
            
            for n in [4, 3, 2, 1]:
                if i + n <= len(words):
                    phrase = ' '.join(words[i:i+n]).lower()
                    if phrase in self._fallback_canonical:
                        result.append(self._fallback_canonical[phrase])
                        i += n
                        matched = True
                        break
            
            if not matched:
                result.append(words[i])
                i += 1
        
        return ' '.join(result)
    
    def get_semantic_types(self, term: str) -> list[str]:
        """Get semantic types for a term (UMLS only)."""
        if self.umls_index:
            return self.umls_index.get_semantic_types(term)
        return []


# Singleton instances
_umls_index: UMLSSynonymIndex | None = None
_hybrid_index: HybridSynonymIndex | None = None


def get_umls_index(force_reload: bool = False) -> UMLSSynonymIndex:
    """Get the global UMLS synonym index."""
    global _umls_index
    
    if _umls_index is None or force_reload:
        _umls_index = UMLSSynonymIndex()
    
    return _umls_index


def get_hybrid_index(
    force_reload: bool = False,
    load_umls: bool = True,
) -> HybridSynonymIndex:
    """Get the global hybrid synonym index."""
    global _hybrid_index
    
    if _hybrid_index is None or force_reload:
        umls_index = None
        if load_umls:
            client = get_opensearch_client()
            if client.connect():
                umls_index = UMLSSynonymIndex(client)
                logger.info("UMLS OpenSearch index ready")
            else:
                logger.warning("UMLS OpenSearch unavailable, using fallback synonyms only")
        
        _hybrid_index = HybridSynonymIndex(umls_index=umls_index)
    
    return _hybrid_index


# Convenience functions
def canonicalize(text: str) -> str:
    """Canonicalize text using the global UMLS index."""
    return get_hybrid_index().canonicalize(text)


def are_synonyms(term1: str, term2: str) -> bool:
    """Check if two terms are UMLS synonyms."""
    return get_hybrid_index().are_synonyms(term1, term2)


def get_synonyms(term: str) -> set[str]:
    """Get all synonyms for a term."""
    return get_hybrid_index().get_synonyms(term)


def search_umls(term: str, limit: int = 10) -> list[UMLSMatch]:
    """Search UMLS for matching concepts."""
    index = get_hybrid_index()
    if index.umls_index:
        return index.umls_index.search_term(term, limit)
    return []
