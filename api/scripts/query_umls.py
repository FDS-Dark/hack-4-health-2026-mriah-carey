#!/usr/bin/env python3
"""
UMLS OpenSearch Query Utility

Simple utility to query UMLS data from OpenSearch for testing and exploration.

Usage:
    python query_umls.py search "diabetes mellitus"
    python query_umls.py lookup C0011849
    python query_umls.py stats

Environment Variables:
    OPENSEARCH_PASSWORD - Required password for OpenSearch admin user
    OPENSEARCH_HOST - OpenSearch host (default: localhost)
    OPENSEARCH_PORT - OpenSearch port (default: 9200)
    OPENSEARCH_USER - OpenSearch user (default: admin)
"""

import os
import sys
import argparse
import json

from opensearchpy import OpenSearch
from dotenv import load_dotenv

load_dotenv()


def create_client() -> OpenSearch:
    """Create OpenSearch client."""
    host = os.getenv("OPENSEARCH_HOST", "localhost")
    port = int(os.getenv("OPENSEARCH_PORT", "9200"))
    user = os.getenv("OPENSEARCH_USER", "admin")
    password = os.getenv("OPENSEARCH_PASSWORD")
    use_ssl = os.getenv("OPENSEARCH_USE_SSL", "true").lower() in ("true", "1", "yes")
    
    if not password:
        print("Error: OPENSEARCH_PASSWORD environment variable is required")
        sys.exit(1)
    
    return OpenSearch(
        hosts=[{"host": host, "port": port}],
        http_auth=(user, password),
        use_ssl=use_ssl,
        verify_certs=False,
        ssl_show_warn=False,
    )


def search_terms(client: OpenSearch, query: str, limit: int = 10) -> None:
    """Search for UMLS concepts by term."""
    print(f"\n=== Searching for: '{query}' ===\n")
    
    # Search in synonyms index for fast term lookup
    body = {
        "query": {
            "bool": {
                "should": [
                    # Exact match (boosted)
                    {"term": {"term_lower": {"value": query.lower(), "boost": 10}}},
                    # Phrase match
                    {"match_phrase": {"term": {"query": query, "boost": 5}}},
                    # Regular match
                    {"match": {"term": {"query": query}}},
                ],
            },
        },
        "size": limit,
        "_source": ["cui", "term", "is_preferred", "source", "semantic_types"],
    }
    
    response = client.search(index="umls_synonyms", body=body)
    hits = response["hits"]["hits"]
    
    if not hits:
        print("No results found.")
        return
    
    print(f"Found {response['hits']['total']['value']} results (showing top {len(hits)}):\n")
    
    for hit in hits:
        src = hit["_source"]
        pref = " [PREFERRED]" if src.get("is_preferred") else ""
        print(f"  CUI: {src['cui']}")
        print(f"  Term: {src['term']}{pref}")
        print(f"  Source: {src.get('source', 'N/A')}")
        print(f"  Semantic Types: {', '.join(src.get('semantic_types', []))}")
        print(f"  Score: {hit['_score']:.2f}")
        print()


def lookup_cui(client: OpenSearch, cui: str) -> None:
    """Look up a specific CUI in the concepts index."""
    print(f"\n=== Looking up CUI: {cui} ===\n")
    
    try:
        response = client.get(index="umls_concepts", id=cui)
        src = response["_source"]
        
        print(f"CUI: {src['cui']}")
        print(f"Preferred Term: {src['preferred_term']}")
        print(f"Languages: {', '.join(src.get('language', []))}")
        print(f"Sources: {', '.join(list(src.get('sources', []))[:10])}...")
        
        print(f"\nSemantic Types:")
        for sty in src.get("semantic_types", []):
            print(f"  - {sty['sty']} ({sty['tui']})")
        
        print(f"\nTerms ({len(src.get('terms', []))} total):")
        for term in list(src.get("terms", []))[:10]:
            print(f"  - {term}")
        if len(src.get("terms", [])) > 10:
            print(f"  ... and {len(src['terms']) - 10} more")
        
        print(f"\nDefinitions ({len(src.get('definitions', []))} total):")
        for defn in src.get("definitions", [])[:3]:
            print(f"  - {defn[:200]}..." if len(defn) > 200 else f"  - {defn}")
        
    except Exception as e:
        print(f"CUI not found: {cui}")
        print(f"Error: {e}")


def show_stats(client: OpenSearch) -> None:
    """Show index statistics."""
    print("\n=== UMLS Index Statistics ===\n")
    
    for index in ["umls_concepts", "umls_synonyms"]:
        try:
            count = client.count(index=index)["count"]
            stats = client.indices.stats(index=index)
            size_bytes = stats["indices"][index]["primaries"]["store"]["size_in_bytes"]
            size_mb = size_bytes / (1024 * 1024)
            
            print(f"{index}:")
            print(f"  Documents: {count:,}")
            print(f"  Size: {size_mb:.1f} MB")
            print()
        except Exception as e:
            print(f"{index}: Not available ({e})")


def main():
    parser = argparse.ArgumentParser(description="Query UMLS data from OpenSearch")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search for concepts by term")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--limit", type=int, default=10, help="Max results")
    
    # Lookup command
    lookup_parser = subparsers.add_parser("lookup", help="Look up a specific CUI")
    lookup_parser.add_argument("cui", help="UMLS Concept Unique Identifier")
    
    # Stats command
    subparsers.add_parser("stats", help="Show index statistics")
    
    args = parser.parse_args()
    client = create_client()
    
    if args.command == "search":
        search_terms(client, args.query, args.limit)
    elif args.command == "lookup":
        lookup_cui(client, args.cui)
    elif args.command == "stats":
        show_stats(client)


if __name__ == "__main__":
    main()
