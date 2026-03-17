"""OpenSearch client for UMLS and medical terminology lookups."""

import os
import logging
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class OpenSearchConfig:
    """Configuration for OpenSearch client."""
    host: str = "localhost"
    port: int = 9200
    user: str = "admin"
    password: str | None = None
    use_ssl: bool = True
    verify_certs: bool = False


class OpenSearchClient:
    """
    Reusable OpenSearch client wrapper.
    
    Provides a single client instance for UMLS lookups and other
    medical terminology searches.
    """
    
    def __init__(self, config: OpenSearchConfig | None = None):
        """
        Initialize OpenSearch client.
        
        Args:
            config: Optional configuration. If not provided, uses environment variables.
        """
        self._client = None
        self._config = config or self._load_config_from_env()
        self._connected = False
    
    @staticmethod
    def _load_config_from_env() -> OpenSearchConfig:
        """Load configuration from environment variables."""
        return OpenSearchConfig(
            host=os.getenv("OPENSEARCH_HOST", "localhost"),
            port=int(os.getenv("OPENSEARCH_PORT", "9200")),
            user=os.getenv("OPENSEARCH_USER", "admin"),
            password=os.getenv("OPENSEARCH_PASSWORD"),
            use_ssl=os.getenv("OPENSEARCH_USE_SSL", "true").lower() in ("true", "1", "yes"),
            verify_certs=os.getenv("OPENSEARCH_VERIFY_CERTS", "false").lower() in ("true", "1", "yes"),
        )
    
    def connect(self) -> bool:
        """
        Establish connection to OpenSearch.
        
        Returns:
            True if connection successful, False otherwise.
        """
        if self._connected and self._client is not None:
            return True
        
        try:
            from opensearchpy import OpenSearch
        except ImportError:
            logger.warning("opensearch-py not installed")
            return False
        
        if not self._config.password:
            logger.warning("OPENSEARCH_PASSWORD not set")
            return False
        
        try:
            self._client = OpenSearch(
                hosts=[{"host": self._config.host, "port": self._config.port}],
                http_auth=(self._config.user, self._config.password),
                use_ssl=self._config.use_ssl,
                verify_certs=self._config.verify_certs,
                ssl_show_warn=False,
            )
            # Test connection
            self._client.info()
            self._connected = True
            logger.info(f"Connected to OpenSearch at {self._config.host}:{self._config.port}")
            return True
        except Exception as e:
            logger.warning(f"Failed to connect to OpenSearch: {e}")
            self._client = None
            self._connected = False
            return False
    
    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._connected and self._client is not None
    
    def search(
        self,
        index: str,
        body: dict[str, Any],
        **kwargs: Any
    ) -> dict[str, Any]:
        """
        Execute a search query.
        
        Args:
            index: Index name to search.
            body: Search query body.
            **kwargs: Additional search parameters.
        
        Returns:
            Search response dictionary.
        
        Raises:
            RuntimeError: If not connected.
        """
        if not self.is_connected:
            if not self.connect():
                raise RuntimeError("OpenSearch client not connected")
        
        return self._client.search(index=index, body=body, **kwargs)
    
    def get(self, index: str, id: str, **kwargs: Any) -> dict[str, Any]:
        """
        Get a document by ID.
        
        Args:
            index: Index name.
            id: Document ID.
            **kwargs: Additional parameters.
        
        Returns:
            Document dictionary.
        
        Raises:
            RuntimeError: If not connected.
        """
        if not self.is_connected:
            if not self.connect():
                raise RuntimeError("OpenSearch client not connected")
        
        return self._client.get(index=index, id=id, **kwargs)
    
    def search_term(
        self,
        term: str,
        index: str = "umls_synonyms",
        limit: int = 10
    ) -> list[dict[str, Any]]:
        """
        Search for a medical term in UMLS.
        
        Args:
            term: Term to search for.
            index: Index to search in.
            limit: Maximum results to return.
        
        Returns:
            List of matching documents.
        """
        if not self.is_connected:
            if not self.connect():
                return []
        
        term_lower = term.lower().strip()
        if not term_lower:
            return []
        
        try:
            body = {
                "query": {
                    "bool": {
                        "should": [
                            {"term": {"term_lower": {"value": term_lower, "boost": 10}}},
                            {"match_phrase": {"term": {"query": term, "boost": 5}}},
                            {"match": {"term": {"query": term, "boost": 1}}},
                        ],
                    },
                },
                "size": limit,
                "_source": ["cui", "term", "is_preferred", "semantic_types"],
            }
            
            response = self.search(index=index, body=body)
            return response.get("hits", {}).get("hits", [])
        except Exception as e:
            logger.warning(f"OpenSearch search error: {e}")
            return []
    
    def get_concept(self, cui: str, index: str = "umls_concepts") -> dict[str, Any] | None:
        """
        Get a UMLS concept by CUI.
        
        Args:
            cui: Concept Unique Identifier.
            index: Index to search in.
        
        Returns:
            Concept document or None if not found.
        """
        try:
            response = self.get(index=index, id=cui)
            return response.get("_source")
        except Exception:
            return None


# Singleton instance
_default_client: OpenSearchClient | None = None


def get_opensearch_client() -> OpenSearchClient:
    """Get or create the default OpenSearch client singleton."""
    global _default_client
    if _default_client is None:
        _default_client = OpenSearchClient()
    return _default_client
