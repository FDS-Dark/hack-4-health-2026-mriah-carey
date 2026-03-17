"""Medical report processing pipeline service with validation and parallel processing."""

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

from clients.gemini import GeminiClient, get_gemini_client
from clients.opensearch import OpenSearchClient, get_opensearch_client
from models.validation import PipelineValidationResult
from services.chunker import ChunkerService
from services.comparison import ComparisonService
from services.simplifier import SimplifierService
from services.verifier import VerifierService, ExaVerifier
from utils.file_handler import safe_json_loads
from validators.orchestrator import ValidationOrchestrator, create_default_orchestrator

logger = logging.getLogger("pipeline")

# Thread pool for CPU-bound validation tasks
_executor: ThreadPoolExecutor | None = None


def get_executor(max_workers: int = 4) -> ThreadPoolExecutor:
    """Get or create the thread pool executor."""
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=max_workers)
    return _executor


@dataclass
class PipelineConfig:
    """Configuration for the pipeline."""
    max_repair_iterations: int = 3
    enable_legacy_comparison: bool = False
    enable_validation: bool = True
    enable_exa_sources: bool = True
    similarity_threshold: float = 0.95
    parallel_chunks: bool = True  # Enable parallel chunk processing
    max_parallel_workers: int = 4  # Max concurrent chunk processing


@dataclass
class ChunkResult:
    """Result of processing a single chunk."""
    original_chunk: str
    simplified_chunk: str
    similarity: float
    sources: str | None
    validation_result: PipelineValidationResult | None
    needs_review: bool
    chunk_index: int = 0  # Track original order
    citations: dict | None = None  # Citation mappings from sources


def process_medical_report(
    text: str,
    config: PipelineConfig | None = None,
    gemini_client: GeminiClient | None = None,
    opensearch_client: OpenSearchClient | None = None,
) -> dict:
    """
    Process a medical report through simplification with validation.
    
    Supports parallel chunk processing for faster throughput.
    
    Architecture:
        report → chunk → [parallel for each chunk]:
                           generate (Gemini)
                           → VALIDATE (hard gates)
                               pass → deliver
                               fail → refine (Gemini repair)
                                    → VALIDATE again (max 2-3 loops)
                                         still fail → "needs review"
    
    Args:
        text: The medical report text to process.
        config: Pipeline configuration options.
        gemini_client: Optional Gemini client for AI operations.
        opensearch_client: Optional OpenSearch client for UMLS lookups.
    
    Returns:
        Dictionary with simplified_report list containing processed chunks.
    """
    config = config or PipelineConfig()
    
    # Check if we're already in an async event loop (e.g., FastAPI)
    try:
        loop = asyncio.get_running_loop()
        # We're in an async context - can't use asyncio.run()
        # Fall back to sequential processing
        logger.info("Detected running event loop - using sequential processing")
        return _process_medical_report_sequential(
            text=text,
            config=config,
            gemini_client=gemini_client,
            opensearch_client=opensearch_client,
        )
    except RuntimeError:
        # No running loop - safe to use asyncio.run() for parallel
        pass
    
    # Use asyncio if parallel is enabled
    if config.parallel_chunks:
        return asyncio.run(_process_medical_report_async(
            text=text,
            config=config,
            gemini_client=gemini_client,
            opensearch_client=opensearch_client,
        ))
    else:
        return _process_medical_report_sequential(
            text=text,
            config=config,
            gemini_client=gemini_client,
            opensearch_client=opensearch_client,
        )


async def _process_medical_report_async(
    text: str,
    config: PipelineConfig,
    gemini_client: GeminiClient | None = None,
    opensearch_client: OpenSearchClient | None = None,
) -> dict:
    """Process medical report with parallel chunk processing."""
    gemini_client = gemini_client or get_gemini_client()
    
    logger.info("="*80)
    logger.info("PIPELINE - Starting medical report processing (PARALLEL MODE)")
    logger.info("="*80)
    logger.info(f"Config: max_repair_iterations={config.max_repair_iterations}")
    logger.info(f"Config: max_parallel_workers={config.max_parallel_workers}")
    logger.info(f"Input text length: {len(text)} characters")
    
    # Initialize shared services
    logger.info("Initializing services...")
    chunker = ChunkerService()
    verifier = VerifierService()
    comparator = ComparisonService()
    
    # Create simplifier (one per chunk to avoid state issues)
    # Pass opensearch for concept extraction
    def create_simplifier():
        return SimplifierService(gemini_client=gemini_client, opensearch_client=opensearch_client)
    
    # Initialize validators
    validator: ValidationOrchestrator | None = None
    if config.enable_validation:
        logger.info("Initializing validation orchestrator...")
        opensearch_client = opensearch_client or get_opensearch_client()
        validator = create_default_orchestrator(opensearch_client=opensearch_client)
    
    # Initialize Exa verifier
    exa_verifier: ExaVerifier | None = None
    if config.enable_exa_sources:
        exa_verifier = ExaVerifier()
    
    logger.info("Services initialized")
    
    # Chunk the text (must be sequential)
    logger.info("\n" + "-"*80)
    logger.info("STEP 1: Chunking text")
    logger.info("-"*80)
    chunked = safe_json_loads(chunker.chunk(text))
    logger.info(f"Created {len(chunked)} chunks")
    
    # Process chunks in parallel
    logger.info("\n" + "-"*80)
    logger.info(f"STEP 2: Processing {len(chunked)} chunks in parallel")
    logger.info("-"*80)
    
    # Create tasks for all chunks
    tasks = []
    for chunk_idx, entry in enumerate(chunked):
        task = _process_chunk_async(
            chunk_idx=chunk_idx,
            original_text=entry["content"],
            create_simplifier=create_simplifier,
            verifier=verifier,
            comparator=comparator,
            validator=validator,
            exa_verifier=exa_verifier,
            config=config,
        )
        tasks.append(task)
    
    # Run all chunk processing concurrently with semaphore limit
    semaphore = asyncio.Semaphore(config.max_parallel_workers)
    
    async def limited_task(task):
        async with semaphore:
            return await task
    
    results = await asyncio.gather(*[limited_task(t) for t in tasks])
    
    # Sort by original chunk index and build output
    results = sorted(results, key=lambda r: r.chunk_index)
    
    simplified_chunks = []
    for chunk_result in results:
        chunk_data = {
            "original_chunk": chunk_result.original_chunk,
            "simplified_chunk": chunk_result.simplified_chunk,
            "similarity": chunk_result.similarity,
            "sources": chunk_result.sources,
            "needs_review": chunk_result.needs_review,
            "validation": chunk_result.validation_result.model_dump() if chunk_result.validation_result else None,
        }
        if chunk_result.citations:
            chunk_data["citations"] = chunk_result.citations
        simplified_chunks.append(chunk_data)
    
    logger.info("\n" + "="*80)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*80)
    logger.info(f"Processed {len(simplified_chunks)} chunks")
    needs_review_count = sum(1 for c in simplified_chunks if c.get("needs_review"))
    if needs_review_count:
        logger.warning(f"{needs_review_count} chunks need manual review")
    else:
        logger.info("All chunks passed validation ✓")
    
    return {"simplified_report": simplified_chunks}


async def _process_chunk_async(
    chunk_idx: int,
    original_text: str,
    create_simplifier: Callable[[], SimplifierService],
    verifier: VerifierService,
    comparator: ComparisonService,
    validator: ValidationOrchestrator | None,
    exa_verifier: ExaVerifier | None,
    config: PipelineConfig,
) -> ChunkResult:
    """Process a single chunk asynchronously."""
    logger.info(f"[Chunk {chunk_idx + 1}] Starting processing ({len(original_text)} chars)")
    
    # Run the blocking chunk processing in thread pool
    loop = asyncio.get_event_loop()
    executor = get_executor(config.max_parallel_workers)
    
    result = await loop.run_in_executor(
        executor,
        _process_chunk_sync,
        chunk_idx,
        original_text,
        create_simplifier(),
        verifier,
        comparator,
        validator,
        exa_verifier,
        config,
    )
    
    logger.info(f"[Chunk {chunk_idx + 1}] Complete - similarity: {result.similarity:.4f}, needs_review: {result.needs_review}")
    return result


def _process_chunk_sync(
    chunk_idx: int,
    original_text: str,
    simplifier: SimplifierService,
    verifier: VerifierService,
    comparator: ComparisonService,
    validator: ValidationOrchestrator | None,
    exa_verifier: ExaVerifier | None,
    config: PipelineConfig,
) -> ChunkResult:
    """Process a single chunk synchronously (runs in thread pool)."""
    # Step 1: Get sources first (if enabled) to use in simplification
    sources = None
    citations = None
    if exa_verifier:
        sources = exa_verifier.search_sources(original_text)
    
    # Step 2: Generate initial simplification with sources
    simplified_response = safe_json_loads(simplifier.simplify(original_text, sources))
    simplified_text = simplified_response["simplified_report"]
    keywords = simplified_response.get("key_words", [])
    citations = simplified_response.get("citations", {})
    
    # Step 3: Legacy comparison checks (if enabled)
    if config.enable_legacy_comparison:
        simplified_text, citations = _run_legacy_comparison_loop(
            original_text=original_text,
            simplified_text=simplified_text,
            keywords=keywords,
            simplifier=simplifier,
            comparator=comparator,
            sources=sources,
        )
    
    # Step 4: Validation loop (if enabled)
    validation_result: PipelineValidationResult | None = None
    needs_review = False
    
    if validator:
        simplified_text, validation_result, needs_review = _run_validation_loop(
            original_text=original_text,
            simplified_text=simplified_text,
            simplifier=simplifier,
            validator=validator,
            max_iterations=config.max_repair_iterations,
        )
    
    # Step 5: Filter sources to only include title, url, and summary for frontend
    filtered_sources = None
    if sources:
        sources_data = json.loads(sources) if isinstance(sources, str) else sources
        filtered_sources = [
            {
                "title": source.get("title", "Untitled"),
                "url": source.get("url", source.get("link", "")),
                "summary": source.get("summary", "No summary available")
            }
            for source in sources_data
        ]
        filtered_sources = json.dumps(filtered_sources)
    
    # Step 6: Compute final similarity
    similarity = verifier.verify_similarity(original_text, simplified_text)
    
    return ChunkResult(
        original_chunk=original_text,
        simplified_chunk=simplified_text,
        similarity=similarity,
        sources=filtered_sources,
        validation_result=validation_result,
        needs_review=needs_review,
        chunk_index=chunk_idx,
        citations=citations,
    )


def _process_medical_report_sequential(
    text: str,
    config: PipelineConfig,
    gemini_client: GeminiClient | None = None,
    opensearch_client: OpenSearchClient | None = None,
) -> dict:
    """Process medical report sequentially (original behavior)."""
    gemini_client = gemini_client or get_gemini_client()
    
    logger.info("="*80)
    logger.info("PIPELINE - Starting medical report processing (SEQUENTIAL MODE)")
    logger.info("="*80)
    
    # Initialize services
    chunker = ChunkerService()
    simplifier = SimplifierService(gemini_client=gemini_client, opensearch_client=opensearch_client)
    verifier = VerifierService()
    comparator = ComparisonService()
    
    # Initialize validators
    validator: ValidationOrchestrator | None = None
    if config.enable_validation:
        opensearch_client = opensearch_client or get_opensearch_client()
        validator = create_default_orchestrator(opensearch_client=opensearch_client)
    
    # Initialize Exa verifier
    exa_verifier: ExaVerifier | None = None
    if config.enable_exa_sources:
        exa_verifier = ExaVerifier()
    
    # Chunk the text
    chunked = safe_json_loads(chunker.chunk(text))
    logger.info(f"Created {len(chunked)} chunks")
    
    simplified_chunks: list[dict] = []
    
    for chunk_idx, entry in enumerate(chunked):
        logger.info(f"\nProcessing chunk {chunk_idx + 1}/{len(chunked)}")
        
        chunk_result = _process_chunk_sync(
            chunk_idx=chunk_idx,
            original_text=entry["content"],
            simplifier=simplifier,
            verifier=verifier,
            comparator=comparator,
            validator=validator,
            exa_verifier=exa_verifier,
            config=config,
        )
        
        chunk_data = {
            "original_chunk": chunk_result.original_chunk,
            "simplified_chunk": chunk_result.simplified_chunk,
            "similarity": chunk_result.similarity,
            "sources": chunk_result.sources,
            "needs_review": chunk_result.needs_review,
            "validation": chunk_result.validation_result.model_dump() if chunk_result.validation_result else None,
        }
        if chunk_result.citations:
            chunk_data["citations"] = chunk_result.citations
        simplified_chunks.append(chunk_data)
    
    logger.info("\n" + "="*80)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*80)
    
    return {"simplified_report": simplified_chunks}


def _run_legacy_comparison_loop(
    original_text: str,
    simplified_text: str,
    keywords: list[str],
    simplifier: SimplifierService,
    comparator: ComparisonService,
    sources: str | None = None,
    max_iterations: int = 5,
) -> tuple[str, dict | None]:
    """Run the legacy comparison-based fix loop."""
    citations = None
    
    for iteration in range(max_iterations):
        entity_missing, entity_extra = comparator.check_entries(simplified_text, keywords)
        number_missing, number_extra = comparator.check_numbers(original_text, simplified_text)
        negation_missing = comparator.check_negations(original_text, simplified_text)
        
        if not entity_missing and not entity_extra and not number_missing and not number_extra and not negation_missing:
            return simplified_text, citations
        
        issues = f"Missing entities: {entity_missing}\nExtra entities: {entity_extra}\nMissing numbers: {number_missing}\nExtra numbers: {number_extra}\nMissing negations: {negation_missing}"
        fixed = safe_json_loads(simplifier.fix(original_text, simplified_text, issues, sources))
        simplified_text = fixed["simplified_report"]
        citations = fixed.get("citations", citations)
    
    return simplified_text, citations


def _run_validation_loop(
    original_text: str,
    simplified_text: str,
    simplifier: SimplifierService,
    validator: ValidationOrchestrator,
    max_iterations: int = 3,
) -> tuple[str, PipelineValidationResult, bool]:
    """Run the validation → refine → validate loop."""
    for iteration in range(1, max_iterations + 1):
        validation_result = validator.validate(
            original_text=original_text,
            simplified_text=simplified_text,
            iteration=iteration,
        )
        
        if validation_result.passed:
            return simplified_text, validation_result, False
        
        if iteration >= max_iterations:
            validation_result.needs_review = True
            return simplified_text, validation_result, True
        
        repair_prompt = validation_result.to_repair_prompt()
        fixed = safe_json_loads(simplifier.repair_with_validation_errors(
            original_text=original_text,
            simplified_text=simplified_text,
            validation_prompt=repair_prompt,
        ))
        simplified_text = fixed["simplified_report"]
    
    validation_result = validator.validate(
        original_text=original_text,
        simplified_text=simplified_text,
        iteration=max_iterations,
    )
    validation_result.needs_review = not validation_result.passed
    
    return simplified_text, validation_result, not validation_result.passed


# Convenience function for async usage
async def process_medical_report_async(
    text: str,
    config: PipelineConfig | None = None,
    gemini_client: GeminiClient | None = None,
    opensearch_client: OpenSearchClient | None = None,
) -> dict:
    """
    Async version of process_medical_report for use in async contexts.
    
    Use this when calling from FastAPI or other async frameworks.
    """
    config = config or PipelineConfig()
    config.parallel_chunks = True
    
    return await _process_medical_report_async(
        text=text,
        config=config,
        gemini_client=gemini_client,
        opensearch_client=opensearch_client,
    )
