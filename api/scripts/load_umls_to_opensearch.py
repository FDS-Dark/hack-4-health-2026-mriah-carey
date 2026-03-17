#!/usr/bin/env python3
"""
UMLS Metathesaurus to OpenSearch Loader

This script loads UMLS Metathesaurus data from RRF files into OpenSearch for
fast medical concept lookups by term name, CUI, or semantic type.

Usage:
    python load_umls_to_opensearch.py --umls-path /path/to/umls/META

Environment Variables:
    OPENSEARCH_PASSWORD - Required password for OpenSearch admin user
    OPENSEARCH_HOST - OpenSearch host (default: localhost)
    OPENSEARCH_PORT - OpenSearch port (default: 9200)
    OPENSEARCH_USER - OpenSearch user (default: admin)
    OPENSEARCH_USE_SSL - Use HTTPS (default: true). Set to "false" for HTTP.

The script creates two indices:
    - umls_concepts: Main concept index with terms, definitions, and semantic types
    - umls_synonyms: Flattened index for fast term/synonym lookups
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Generator
from dataclasses import dataclass, field
from collections import defaultdict

from opensearchpy import OpenSearch, helpers
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# UMLS RRF File Schemas
# =============================================================================

# MRCONSO.RRF columns: Concept names and sources
MRCONSO_COLUMNS = [
    "CUI",      # Unique identifier for concept
    "LAT",      # Language of term
    "TS",       # Term status
    "LUI",      # Unique identifier for term
    "STT",      # String type
    "SUI",      # Unique identifier for string
    "ISPREF",   # Indicates whether AUI is preferred
    "AUI",      # Unique identifier for atom
    "SAUI",     # Source asserted atom identifier
    "SCUI",     # Source asserted concept identifier
    "SDUI",     # Source asserted descriptor identifier
    "SAB",      # Source abbreviation
    "TTY",      # Term type in source
    "CODE",     # Unique identifier or code for string in source
    "STR",      # String (the actual term text)
    "SRL",      # Source restriction level
    "SUPPRESS", # Suppressible flag
    "CVF",      # Content view flag
]

# MRDEF.RRF columns: Definitions
MRDEF_COLUMNS = [
    "CUI",      # Unique identifier for concept
    "AUI",      # Unique identifier for atom
    "ATUI",     # Unique identifier for attribute
    "SATUI",    # Source asserted attribute identifier
    "SAB",      # Source abbreviation
    "DEF",      # Definition text
    "SUPPRESS", # Suppressible flag
    "CVF",      # Content view flag
]

# MRSTY.RRF columns: Semantic types
MRSTY_COLUMNS = [
    "CUI",      # Unique identifier for concept
    "TUI",      # Unique identifier of semantic type
    "STN",      # Semantic type tree number
    "STY",      # Semantic type name
    "ATUI",     # Unique identifier for attribute
    "CVF",      # Content view flag
]

# MRREL.RRF columns: Related concepts (optional, for relationships)
MRREL_COLUMNS = [
    "CUI1",     # First concept
    "AUI1",     # First atom
    "STYPE1",   # Source type 1
    "REL",      # Relationship label
    "CUI2",     # Second concept
    "AUI2",     # Second atom
    "STYPE2",   # Source type 2
    "RELA",     # Additional relationship label
    "RUI",      # Unique identifier for relationship
    "SRUI",     # Source relationship identifier
    "SAB",      # Source abbreviation
    "SL",       # Source of relationship labels
    "RG",       # Relationship group
    "DIR",      # Directionality flag
    "SUPPRESS", # Suppressible flag
    "CVF",      # Content view flag
]


# =============================================================================
# OpenSearch Index Mappings
# =============================================================================

CONCEPTS_INDEX_MAPPING = {
    "settings": {
        "index": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
        },
        "analysis": {
            "analyzer": {
                "medical_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "asciifolding"],
                },
            },
        },
    },
    "mappings": {
        "properties": {
            "cui": {"type": "keyword"},
            "preferred_term": {
                "type": "text",
                "analyzer": "medical_analyzer",
                "fields": {"keyword": {"type": "keyword"}},
            },
            "terms": {
                "type": "text",
                "analyzer": "medical_analyzer",
            },
            "definitions": {"type": "text", "analyzer": "medical_analyzer"},
            "semantic_types": {
                "type": "nested",
                "properties": {
                    "tui": {"type": "keyword"},
                    "sty": {"type": "keyword"},
                },
            },
            "sources": {"type": "keyword"},
            "language": {"type": "keyword"},
        },
    },
}

SYNONYMS_INDEX_MAPPING = {
    "settings": {
        "index": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
        },
        "analysis": {
            "analyzer": {
                "medical_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "asciifolding"],
                },
            },
        },
    },
    "mappings": {
        "properties": {
            "cui": {"type": "keyword"},
            "term": {
                "type": "text",
                "analyzer": "medical_analyzer",
                "fields": {"keyword": {"type": "keyword"}},
            },
            "term_lower": {"type": "keyword"},  # For exact matching
            "is_preferred": {"type": "boolean"},
            "language": {"type": "keyword"},
            "source": {"type": "keyword"},
            "term_type": {"type": "keyword"},
            "semantic_types": {"type": "keyword"},  # TUI codes
        },
    },
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class UMLSConcept:
    """Aggregated UMLS concept with all terms, definitions, and semantic types."""
    cui: str
    preferred_term: str = ""
    terms: set = field(default_factory=set)
    definitions: list = field(default_factory=list)
    semantic_types: list = field(default_factory=list)  # List of {tui, sty}
    sources: set = field(default_factory=set)
    languages: set = field(default_factory=set)


# =============================================================================
# RRF File Parser
# =============================================================================

def parse_rrf_file(
    file_path: Path,
    columns: list[str],
    encoding: str = "utf-8",
) -> Generator[dict, None, None]:
    """
    Parse a UMLS RRF file and yield dictionaries for each row.
    
    RRF files are pipe-delimited with a trailing pipe.
    """
    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return
    
    logger.info(f"Parsing {file_path.name}...")
    line_count = 0
    
    with open(file_path, "r", encoding=encoding, errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            
            # RRF files have trailing pipe, so we strip it and split
            parts = line.rstrip("|").split("|")
            
            # Create dict from columns and values
            row = {}
            for i, col in enumerate(columns):
                row[col] = parts[i] if i < len(parts) else ""
            
            yield row
            line_count += 1
            
            if line_count % 1_000_000 == 0:
                logger.info(f"  Processed {line_count:,} lines...")
    
    logger.info(f"  Finished parsing {line_count:,} lines from {file_path.name}")


# =============================================================================
# Data Aggregation
# =============================================================================

def load_semantic_types(meta_path: Path) -> dict[str, list[dict]]:
    """Load semantic types from MRSTY.RRF indexed by CUI."""
    sty_file = meta_path / "MRSTY.RRF"
    cui_to_sty = defaultdict(list)
    
    for row in parse_rrf_file(sty_file, MRSTY_COLUMNS):
        cui_to_sty[row["CUI"]].append({
            "tui": row["TUI"],
            "sty": row["STY"],
        })
    
    return dict(cui_to_sty)


def load_definitions(meta_path: Path) -> dict[str, list[str]]:
    """Load definitions from MRDEF.RRF indexed by CUI."""
    def_file = meta_path / "MRDEF.RRF"
    cui_to_def = defaultdict(list)
    
    for row in parse_rrf_file(def_file, MRDEF_COLUMNS):
        if row["SUPPRESS"] != "Y":  # Skip suppressed definitions
            definition = row["DEF"]
            if definition and definition not in cui_to_def[row["CUI"]]:
                cui_to_def[row["CUI"]].append(definition)
    
    return dict(cui_to_def)


def aggregate_concepts(
    meta_path: Path,
    semantic_types: dict[str, list[dict]],
    definitions: dict[str, list[str]],
    languages: set[str] | None = None,
) -> Generator[UMLSConcept, None, None]:
    """
    Aggregate MRCONSO.RRF rows into UMLSConcept objects.
    
    Args:
        meta_path: Path to META directory
        semantic_types: Pre-loaded semantic types by CUI
        definitions: Pre-loaded definitions by CUI
        languages: Set of language codes to include (None = all)
    """
    conso_file = meta_path / "MRCONSO.RRF"
    
    # Group all rows by CUI
    concepts = {}
    
    for row in parse_rrf_file(conso_file, MRCONSO_COLUMNS):
        cui = row["CUI"]
        lang = row["LAT"]
        
        # Filter by language if specified
        if languages and lang not in languages:
            continue
        
        # Skip suppressed entries
        if row["SUPPRESS"] == "Y":
            continue
        
        # Initialize concept if new
        if cui not in concepts:
            concepts[cui] = UMLSConcept(
                cui=cui,
                semantic_types=semantic_types.get(cui, []),
                definitions=definitions.get(cui, []),
            )
        
        concept = concepts[cui]
        term = row["STR"]
        
        # Track all terms
        concept.terms.add(term)
        concept.sources.add(row["SAB"])
        concept.languages.add(lang)
        
        # Set preferred term (P = preferred, Y = is preferred for this AUI)
        if row["TS"] == "P" and row["ISPREF"] == "Y" and lang == "ENG":
            if not concept.preferred_term:
                concept.preferred_term = term
    
    # Yield all concepts
    for concept in concepts.values():
        # Fallback preferred term if none found
        if not concept.preferred_term and concept.terms:
            concept.preferred_term = next(iter(concept.terms))
        yield concept


# =============================================================================
# OpenSearch Operations
# =============================================================================

def create_opensearch_client() -> OpenSearch:
    """Create and return an OpenSearch client using environment variables."""
    host = os.getenv("OPENSEARCH_HOST", "localhost")
    port = int(os.getenv("OPENSEARCH_PORT", "9200"))
    user = os.getenv("OPENSEARCH_USER", "admin")
    password = os.getenv("OPENSEARCH_PASSWORD")
    use_ssl = os.getenv("OPENSEARCH_USE_SSL", "true").lower() in ("true", "1", "yes")
    
    if not password:
        raise ValueError(
            "OPENSEARCH_PASSWORD environment variable is required. "
            "Set it with: export OPENSEARCH_PASSWORD=your_password"
        )
    
    protocol = "https" if use_ssl else "http"
    logger.info(f"Connecting to OpenSearch at {protocol}://{host}:{port}")
    
    client = OpenSearch(
        hosts=[{"host": host, "port": port}],
        http_auth=(user, password),
        use_ssl=use_ssl,
        verify_certs=False,  # For local development with self-signed certs
        ssl_show_warn=False,
    )
    
    # Test connection
    info = client.info()
    logger.info(f"Connected to OpenSearch cluster: {info['cluster_name']}")
    
    return client


def create_indices(client: OpenSearch, force: bool = False) -> None:
    """Create the UMLS indices in OpenSearch."""
    indices = [
        ("umls_concepts", CONCEPTS_INDEX_MAPPING),
        ("umls_synonyms", SYNONYMS_INDEX_MAPPING),
    ]
    
    for index_name, mapping in indices:
        if client.indices.exists(index=index_name):
            if force:
                logger.info(f"Deleting existing index: {index_name}")
                client.indices.delete(index=index_name)
            else:
                logger.info(f"Index already exists: {index_name}")
                continue
        
        logger.info(f"Creating index: {index_name}")
        client.indices.create(index=index_name, body=mapping)


def generate_concept_documents(
    concepts: Generator[UMLSConcept, None, None],
) -> Generator[dict, None, None]:
    """Generate OpenSearch bulk documents for concepts index."""
    for concept in concepts:
        yield {
            "_index": "umls_concepts",
            "_id": concept.cui,
            "_source": {
                "cui": concept.cui,
                "preferred_term": concept.preferred_term,
                "terms": list(concept.terms),
                "definitions": concept.definitions[:5],  # Limit definitions
                "semantic_types": concept.semantic_types,
                "sources": list(concept.sources),
                "language": list(concept.languages),
            },
        }


def generate_synonym_documents(
    meta_path: Path,
    semantic_types: dict[str, list[dict]],
    languages: set[str] | None = None,
) -> Generator[dict, None, None]:
    """Generate OpenSearch bulk documents for synonyms index."""
    conso_file = meta_path / "MRCONSO.RRF"
    seen = set()  # Track (cui, term_lower) to avoid duplicates
    
    for row in parse_rrf_file(conso_file, MRCONSO_COLUMNS):
        cui = row["CUI"]
        lang = row["LAT"]
        term = row["STR"]
        
        # Filter by language if specified
        if languages and lang not in languages:
            continue
        
        # Skip suppressed entries
        if row["SUPPRESS"] == "Y":
            continue
        
        # Deduplicate by (CUI, lowercase term)
        term_lower = term.lower()
        key = (cui, term_lower)
        if key in seen:
            continue
        seen.add(key)
        
        # Get semantic type TUIs for this concept
        stys = semantic_types.get(cui, [])
        tuis = [s["tui"] for s in stys]
        
        yield {
            "_index": "umls_synonyms",
            "_source": {
                "cui": cui,
                "term": term,
                "term_lower": term_lower,
                "is_preferred": row["TS"] == "P" and row["ISPREF"] == "Y",
                "language": lang,
                "source": row["SAB"],
                "term_type": row["TTY"],
                "semantic_types": tuis,
            },
        }


def bulk_index(
    client: OpenSearch,
    documents: Generator[dict, None, None],
    chunk_size: int = 5000,
) -> int:
    """Bulk index documents into OpenSearch with progress logging."""
    total = 0
    success_count = 0
    
    for ok, result in helpers.streaming_bulk(
        client,
        documents,
        chunk_size=chunk_size,
        raise_on_error=False,
        raise_on_exception=False,
    ):
        total += 1
        if ok:
            success_count += 1
        else:
            logger.warning(f"Failed to index document: {result}")
        
        if total % 100_000 == 0:
            logger.info(f"  Indexed {total:,} documents ({success_count:,} successful)")
    
    logger.info(f"  Finished indexing {total:,} documents ({success_count:,} successful)")
    return success_count


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Load UMLS Metathesaurus into OpenSearch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--umls-path",
        type=Path,
        required=True,
        help="Path to UMLS META directory (containing MRCONSO.RRF, etc.)",
    )
    parser.add_argument(
        "--languages",
        type=str,
        default="ENG",
        help="Comma-separated language codes to include (default: ENG). Use 'all' for all languages.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and recreate indices if they exist",
    )
    parser.add_argument(
        "--concepts-only",
        action="store_true",
        help="Only load the concepts index (skip synonyms)",
    )
    parser.add_argument(
        "--synonyms-only",
        action="store_true",
        help="Only load the synonyms index (skip concepts)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5000,
        help="Bulk indexing chunk size (default: 5000)",
    )
    
    args = parser.parse_args()
    
    # Validate UMLS path
    meta_path = args.umls_path
    if not meta_path.exists():
        logger.error(f"UMLS path does not exist: {meta_path}")
        sys.exit(1)
    
    # Check for required files
    required_files = ["MRCONSO.RRF", "MRSTY.RRF"]
    for filename in required_files:
        if not (meta_path / filename).exists():
            logger.error(f"Required file not found: {meta_path / filename}")
            sys.exit(1)
    
    # Parse languages
    languages = None
    if args.languages.lower() != "all":
        languages = set(args.languages.upper().split(","))
        logger.info(f"Filtering to languages: {languages}")
    
    try:
        # Connect to OpenSearch
        client = create_opensearch_client()
        
        # Create indices
        create_indices(client, force=args.force)
        
        # Load semantic types and definitions (needed for both indices)
        logger.info("Loading semantic types...")
        semantic_types = load_semantic_types(meta_path)
        logger.info(f"  Loaded semantic types for {len(semantic_types):,} concepts")
        
        # Load concepts index
        if not args.synonyms_only:
            logger.info("Loading definitions...")
            definitions = load_definitions(meta_path)
            logger.info(f"  Loaded definitions for {len(definitions):,} concepts")
            
            logger.info("Indexing concepts...")
            concepts = aggregate_concepts(
                meta_path, semantic_types, definitions, languages
            )
            docs = generate_concept_documents(concepts)
            bulk_index(client, docs, chunk_size=args.chunk_size)
        
        # Load synonyms index
        if not args.concepts_only:
            logger.info("Indexing synonyms...")
            docs = generate_synonym_documents(meta_path, semantic_types, languages)
            bulk_index(client, docs, chunk_size=args.chunk_size)
        
        # Refresh indices
        logger.info("Refreshing indices...")
        if not args.synonyms_only:
            client.indices.refresh(index="umls_concepts")
        if not args.concepts_only:
            client.indices.refresh(index="umls_synonyms")
        
        # Print stats
        logger.info("Done! Index statistics:")
        if not args.synonyms_only:
            count = client.count(index="umls_concepts")["count"]
            logger.info(f"  umls_concepts: {count:,} documents")
        if not args.concepts_only:
            count = client.count(index="umls_synonyms")["count"]
            logger.info(f"  umls_synonyms: {count:,} documents")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
