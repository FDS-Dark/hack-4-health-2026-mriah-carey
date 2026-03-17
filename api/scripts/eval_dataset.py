#!/usr/bin/env python3
"""
Run evaluations on the merged_plain_language_dataset.csv.

This script:
1. Takes original texts from the dataset
2. Runs them through the simplification pipeline (with parallel processing)
3. Evaluates the GENERATED outputs against the REFERENCE plain language texts
4. Exports per-sample stats for scatter plots

Usage:
    # Evaluate a sample of the dataset (default: 10 samples)
    python scripts/eval_dataset.py evaluate -n 10
    
    # Evaluate with parallel processing (faster)
    python scripts/eval_dataset.py evaluate -n 50 --parallel 4
    
    # Export detailed stats for scatter plots
    python scripts/eval_dataset.py evaluate -n 20 --stats-csv results_stats.csv
    
    # Filter by source dataset
    python scripts/eval_dataset.py evaluate -n 20 --source cochrane
"""

import csv
import json
import logging
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import typer

# Increase CSV field size limit for large text fields (Windows-compatible)
try:
    csv.field_size_limit(2**31 - 1)  # Windows max
except OverflowError:
    csv.field_size_limit(sys.maxsize)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.evaluation_service import EvaluationOrchestrator, EvaluationConfig
from models.evaluation import DatasetEvaluation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S'
)

# Reduce noise
for name in ["urllib3", "httpx", "transformers", "sentence_transformers", "filelock", "opensearch"]:
    logging.getLogger(name).setLevel(logging.WARNING)

logger = logging.getLogger("eval_dataset")

app = typer.Typer(
    name="eval-dataset",
    help="Run evaluations on the merged plain language dataset.",
)

DEFAULT_DATASET_PATH = Path(__file__).parent.parent.parent / "merged_plain_language_dataset.csv"
REPORTS_DATASET_PATH = Path(__file__).parent.parent.parent / "ReportsDATASET.csv"


@dataclass
class SampleStats:
    """Per-sample statistics for scatter plot analysis."""
    sample_id: str
    source_dataset: str = ""
    
    # Input stats
    original_chars: int = 0
    original_words: int = 0
    original_fkgl: float = 0.0
    original_smog: float = 0.0
    original_jargon_density: float = 0.0
    
    # Output stats
    simplified_chars: int = 0
    simplified_words: int = 0
    simplified_fkgl: float = 0.0
    simplified_smog: float = 0.0
    simplified_jargon_density: float = 0.0
    
    # Delta stats
    fkgl_delta: float = 0.0
    smog_delta: float = 0.0
    jargon_reduction: float = 0.0
    expansion_ratio: float = 0.0
    
    # Quality stats
    concept_recall: float = 0.0
    critical_concept_recall: float = 0.0
    measurement_preservation: float = 0.0
    simplification_score: float = 0.0
    
    # Pipeline stats
    latency_ms: float = 0.0
    chunk_count: int = 0
    validation_iterations: int = 1
    first_pass_success: bool = True
    needs_review: bool = False
    
    # Errors
    error: str = ""


def load_dataset(
    path: Path,
    max_samples: int = 10,
    source_filter: Optional[str] = None,
    split_filter: Optional[str] = None,
    seed: int = 42,
) -> list[dict]:
    """
    Load samples from a dataset. Supports multiple formats:
    
    1. Plain language dataset: has 'original_text' and 'plain_language_text' columns
    2. Reports dataset: has just 'Text' column (no ground truth)
    
    Returns:
        List of dicts with 'original_text' and optionally 'reference_text'.
    """
    logger.info(f"Loading dataset from {path}")
    
    samples = []
    
    with open(path, 'r', encoding='utf-8-sig') as f:  # utf-8-sig handles BOM
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        
        # Detect dataset format
        if 'Text' in fieldnames and 'original_text' not in fieldnames:
            # ReportsDATASET format - just raw medical reports
            logger.info("Detected ReportsDATASET format (raw reports, no ground truth)")
            for idx, row in enumerate(reader):
                text = row.get('Text', '').strip()
                if not text or len(text) < 50:  # Skip empty/tiny entries
                    continue
                
                samples.append({
                    'sample_id': str(idx),
                    'original_text': text,
                    'reference_text': None,  # No ground truth available
                    'source_dataset': 'reports',
                    'split': 'unknown',
                })
        else:
            # Plain language dataset format
            logger.info("Detected plain language dataset format (with ground truth)")
            for row in reader:
                # Apply filters
                if source_filter and row.get('source_dataset', '').lower() != source_filter.lower():
                    continue
                if split_filter and row.get('split', '').lower() != split_filter.lower():
                    continue
                
                # Skip empty rows
                original = row.get('original_text', '').strip()
                reference = row.get('plain_language_text', '').strip()
                
                if not original or not reference:
                    continue
                
                samples.append({
                    'sample_id': row.get('index', str(len(samples))),
                    'original_text': original,
                    'reference_text': reference,  # Ground truth
                    'source_dataset': row.get('source_dataset', 'unknown'),
                    'split': row.get('split', 'unknown'),
                })
    
    logger.info(f"Loaded {len(samples)} total samples")
    
    # Sample if needed
    if max_samples > 0 and len(samples) > max_samples:
        random.seed(seed)
        samples = random.sample(samples, max_samples)
        logger.info(f"Randomly sampled {max_samples} samples")
    
    return samples


def run_pipeline_with_stats(sample: dict) -> tuple[dict, SampleStats]:
    """
    Run the simplification pipeline on a sample and collect detailed stats.
    
    Returns:
        Tuple of (eval_sample dict, SampleStats)
    """
    from pipeline import process_medical_report, PipelineConfig
    from evaluators.simplification import _compute_readability, _compute_jargon_density
    
    sample_id = sample['sample_id']
    original_text = sample['original_text']
    
    # Initialize stats
    stats = SampleStats(
        sample_id=sample_id,
        source_dataset=sample.get('source_dataset', ''),
        original_chars=len(original_text),
        original_words=len(original_text.split()),
    )
    
    # Compute original readability
    try:
        orig_read = _compute_readability(original_text)
        stats.original_fkgl = orig_read['fkgl']
        stats.original_smog = orig_read['smog']
        stats.original_jargon_density = _compute_jargon_density(original_text)
    except Exception:
        pass
    
    # Run pipeline
    config = PipelineConfig(
        max_repair_iterations=2,
        enable_legacy_comparison=False,
        enable_validation=True,
        enable_exa_sources=False,
        parallel_chunks=False,
    )
    
    start_time = time.time()
    
    try:
        result = process_medical_report(original_text, config=config)
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Combine all simplified chunks
        simplified_chunks = result.get("simplified_report", [])
        simplified_text = "\n\n".join(
            chunk.get("simplified_chunk", "") 
            for chunk in simplified_chunks
        )
        
        # Collect pipeline stats
        stats.latency_ms = elapsed_ms
        stats.chunk_count = len(simplified_chunks)
        stats.needs_review = any(c.get("needs_review") for c in simplified_chunks)
        stats.first_pass_success = not stats.needs_review
        
        # Count validation iterations from chunks
        total_iterations = sum(
            c.get("validation", {}).get("iteration", 1) if c.get("validation") else 1
            for c in simplified_chunks
        )
        stats.validation_iterations = total_iterations // max(1, len(simplified_chunks))
        
        # Compute simplified readability
        stats.simplified_chars = len(simplified_text)
        stats.simplified_words = len(simplified_text.split())
        
        try:
            simp_read = _compute_readability(simplified_text)
            stats.simplified_fkgl = simp_read['fkgl']
            stats.simplified_smog = simp_read['smog']
            stats.simplified_jargon_density = _compute_jargon_density(simplified_text)
            
            # Compute deltas
            stats.fkgl_delta = stats.simplified_fkgl - stats.original_fkgl
            stats.smog_delta = stats.simplified_smog - stats.original_smog
            stats.jargon_reduction = stats.original_jargon_density - stats.simplified_jargon_density
            stats.expansion_ratio = stats.simplified_words / max(1, stats.original_words)
        except Exception:
            pass
        
        # Build eval sample
        eval_sample = {
            'sample_id': sample_id,
            'original_text': original_text,
            'simplified_text': simplified_text,
            'reference_text': sample.get('reference_text'),
            'source_dataset': sample.get('source_dataset'),
            'stats': stats,
        }
        
        return eval_sample, stats
        
    except Exception as e:
        stats.error = str(e)
        stats.latency_ms = (time.time() - start_time) * 1000
        return None, stats


def process_samples_parallel(
    samples: list[dict],
    max_workers: int = 4,
    progress_callback=None,
) -> tuple[list[dict], list[SampleStats]]:
    """
    Process multiple samples through the pipeline in parallel.
    
    Returns:
        Tuple of (eval_samples, all_stats)
    """
    eval_samples = []
    all_stats = []
    completed = 0
    total = len(samples)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_sample = {
            executor.submit(run_pipeline_with_stats, sample): sample
            for sample in samples
        }
        
        # Process as they complete
        for future in as_completed(future_to_sample):
            sample = future_to_sample[future]
            completed += 1
            
            try:
                eval_sample, stats = future.result()
                all_stats.append(stats)
                
                if eval_sample:
                    eval_samples.append(eval_sample)
                    logger.info(
                        f"[{completed}/{total}] Sample {stats.sample_id}: "
                        f"FKGL {stats.original_fkgl:.1f}→{stats.simplified_fkgl:.1f} "
                        f"({stats.fkgl_delta:+.1f}) in {stats.latency_ms:.0f}ms"
                    )
                else:
                    logger.error(f"[{completed}/{total}] Sample {stats.sample_id}: {stats.error}")
                    
            except Exception as e:
                logger.error(f"[{completed}/{total}] Sample {sample['sample_id']}: {e}")
                all_stats.append(SampleStats(
                    sample_id=sample['sample_id'],
                    error=str(e),
                ))
    
    return eval_samples, all_stats


def export_stats_csv(stats: list[SampleStats], path: Path):
    """Export stats to CSV for scatter plot analysis."""
    if not stats:
        return
    
    fieldnames = list(asdict(stats[0]).keys())
    
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for stat in stats:
            writer.writerow(asdict(stat))
    
    logger.info(f"Exported {len(stats)} sample stats to {path}")


def parse_evaluators(evaluators_str: str) -> set[str]:
    """Parse comma-separated evaluator names."""
    valid = {"readability", "semantic", "concept", "nli", "coverage", "ops"}
    requested = {e.strip().lower() for e in evaluators_str.split(",") if e.strip()}
    
    invalid = requested - valid
    if invalid:
        raise typer.BadParameter(f"Unknown evaluators: {invalid}. Valid: {valid}")
    
    return requested


@app.command()
def evaluate(
    dataset_path: Optional[Path] = typer.Option(
        None,
        "--dataset", "-d",
        help="Path to the CSV dataset file.",
    ),
    samples: int = typer.Option(
        10,
        "--samples", "-n",
        help="Number of samples to evaluate.",
    ),
    parallel: int = typer.Option(
        1,
        "--parallel", "-p",
        help="Number of parallel workers (1 = sequential).",
    ),
    source_filter: Optional[str] = typer.Option(
        None,
        "--source", "-s",
        help="Filter by source_dataset (e.g., 'cochrane', 'CELLS').",
    ),
    split_filter: Optional[str] = typer.Option(
        None,
        "--split",
        help="Filter by split (e.g., 'train', 'test').",
    ),
    evaluators: Optional[str] = typer.Option(
        None,
        "--evaluators", "-e",
        help="Comma-separated list of evaluators (default: all fast ones).",
    ),
    semantic_model: str = typer.Option(
        "pubmedbert",
        "--semantic-model",
        help="Semantic model: 'pubmedbert' or 'bioclinicalbert'.",
    ),
    output_path: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Path to save evaluation results JSON.",
    ),
    stats_csv: Optional[Path] = typer.Option(
        None,
        "--stats-csv",
        help="Path to export per-sample stats CSV for scatter plots.",
    ),
    experiment_name: str = typer.Option(
        "pipeline-evals",
        "--experiment",
        help="MLflow experiment name.",
    ),
    run_name: Optional[str] = typer.Option(
        None,
        "--run-name",
        help="Custom name for the MLflow run.",
    ),
    model_version: str = typer.Option(
        "v1",
        "--model-version", "-v",
        help="Model version identifier.",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed for sampling.",
    ),
    fast: bool = typer.Option(
        True,
        "--fast/--full",
        help="Use fast evaluators only (skip semantic, NLI).",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-V",
        help="Enable verbose logging.",
    ),
):
    """
    Evaluate the pipeline on the plain language dataset.
    
    For each sample:
    1. Run original_text through the simplification pipeline
    2. Compare generated output against reference_text (ground truth)
    3. Compute evaluation metrics
    4. Export detailed stats for analysis
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Use default path if not specified
    path = dataset_path or DEFAULT_DATASET_PATH
    
    if not path.exists():
        logger.error(f"Dataset not found: {path}")
        raise typer.Exit(1)
    
    # Load samples
    dataset_samples = load_dataset(
        path=path,
        max_samples=samples,
        source_filter=source_filter,
        split_filter=split_filter,
        seed=seed,
    )
    
    if not dataset_samples:
        logger.error("No samples loaded!")
        raise typer.Exit(1)
    
    # Determine which evaluators to use
    enabled_evaluators = None
    if evaluators:
        enabled_evaluators = parse_evaluators(evaluators)
    elif fast:
        enabled_evaluators = {"readability", "coverage", "ops"}
    
    # Build dataset name
    dataset_name = "pipeline_eval"
    if source_filter:
        dataset_name += f"_{source_filter}"
    if split_filter:
        dataset_name += f"_{split_filter}"
    
    # Process samples through pipeline
    logger.info(f"Processing {len(dataset_samples)} samples through pipeline...")
    if parallel > 1:
        logger.info(f"Using {parallel} parallel workers")
        eval_samples, all_stats = process_samples_parallel(
            dataset_samples,
            max_workers=parallel,
        )
    else:
        # Sequential processing
        eval_samples = []
        all_stats = []
        for i, sample in enumerate(dataset_samples):
            logger.info(f"[{i+1}/{len(dataset_samples)}] Processing sample {sample['sample_id']}...")
            eval_sample, stats = run_pipeline_with_stats(sample)
            all_stats.append(stats)
            
            if eval_sample:
                eval_samples.append(eval_sample)
                logger.info(
                    f"    FKGL {stats.original_fkgl:.1f}→{stats.simplified_fkgl:.1f} "
                    f"({stats.fkgl_delta:+.1f}) in {stats.latency_ms:.0f}ms"
                )
            else:
                logger.error(f"    Failed: {stats.error}")
    
    # Export stats CSV if requested
    if stats_csv:
        export_stats_csv(all_stats, stats_csv)
    
    if not eval_samples:
        logger.error("No samples processed successfully!")
        raise typer.Exit(1)
    
    logger.info(f"\nSuccessfully processed {len(eval_samples)}/{len(dataset_samples)} samples")
    
    # Create evaluation config
    config = EvaluationConfig(
        enable_readability=(enabled_evaluators is None or "readability" in enabled_evaluators),
        enable_simplification=True,  # Always enable for before/after comparison
        enable_semantic=(enabled_evaluators is None or "semantic" in enabled_evaluators),
        enable_concept=(enabled_evaluators is None or "concept" in enabled_evaluators),
        enable_nli=(enabled_evaluators is None or "nli" in enabled_evaluators),
        enable_coverage=(enabled_evaluators is None or "coverage" in enabled_evaluators),
        enable_ops=(enabled_evaluators is None or "ops" in enabled_evaluators),
        semantic_model=semantic_model,
        experiment_name=experiment_name,
        dataset_name=dataset_name,
        model_version=model_version,
    )
    
    # Create orchestrator
    logger.info("Initializing evaluation orchestrator...")
    orchestrator = EvaluationOrchestrator(config=config)
    
    # Record ops metrics from stats
    ops_evaluator = orchestrator.evaluators.get("ops")
    if ops_evaluator:
        for stats in all_stats:
            if not stats.error:
                ops_evaluator.record_run(
                    latency_ms=stats.latency_ms,
                    input_tokens=stats.original_words,
                    output_tokens=stats.simplified_words,
                    retry_count=max(0, stats.validation_iterations - 1),
                    refinement_count=0,
                    first_pass_success=stats.first_pass_success,
                )
    
    # Prepare samples for evaluation (strip stats)
    eval_samples_clean = [
        {k: v for k, v in s.items() if k != 'stats'}
        for s in eval_samples
    ]
    
    # Run evaluations
    logger.info("Running evaluations on generated outputs...")
    
    if not run_name:
        run_name = f"{dataset_name}-{model_version}-n{len(eval_samples)}"
    
    results = orchestrator.evaluate_dataset(
        samples=eval_samples_clean,
        run_name=run_name,
        tags={
            "dataset": dataset_name,
            "model_version": model_version,
            "sample_count": str(len(eval_samples)),
            "source_filter": source_filter or "all",
            "split_filter": split_filter or "all",
            "eval_type": "pipeline",
            "parallel_workers": str(parallel),
        },
    )
    
    # Print summary
    _print_summary(results)
    
    # Print quick stats summary
    _print_stats_summary(all_stats)
    
    # Save results if output path provided
    if output_path:
        output_results = results.model_dump(mode="json")
        # Truncate large text fields
        for sample in output_results.get("samples", []):
            for key in ["original_text", "simplified_text", "reference_text"]:
                if sample.get(key) and len(sample[key]) > 500:
                    sample[key] = sample[key][:500] + "..."
        
        with open(output_path, "w") as f:
            json.dump(output_results, f, indent=2)
        logger.info(f"Results saved to {output_path}")
    
    logger.info(f"MLflow run ID: {results.run_id}")
    logger.info("Evaluation complete!")


@app.command()
def baseline(
    dataset_path: Optional[Path] = typer.Option(
        None,
        "--dataset", "-d",
        help="Path to the CSV dataset file.",
    ),
    samples: int = typer.Option(
        100,
        "--samples", "-n",
        help="Number of samples to evaluate.",
    ),
    source_filter: Optional[str] = typer.Option(
        None,
        "--source", "-s",
        help="Filter by source_dataset.",
    ),
    output_path: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Path to save evaluation results JSON.",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed for sampling.",
    ),
):
    """
    Evaluate the REFERENCE texts quality (baseline metrics).
    
    This measures the quality of the ground truth simplifications
    without running the pipeline. Useful for understanding target metrics.
    """
    path = dataset_path or DEFAULT_DATASET_PATH
    
    if not path.exists():
        logger.error(f"Dataset not found: {path}")
        raise typer.Exit(1)
    
    # Load samples
    dataset_samples = load_dataset(
        path=path,
        max_samples=samples,
        source_filter=source_filter,
        seed=seed,
    )
    
    if not dataset_samples:
        logger.error("No samples loaded!")
        raise typer.Exit(1)
    
    # For baseline, we evaluate the reference text itself
    eval_samples = [
        {
            'sample_id': s['sample_id'],
            'original_text': s['original_text'],
            'simplified_text': s['reference_text'],  # Evaluate reference quality
            'reference_text': s['reference_text'],
        }
        for s in dataset_samples
    ]
    
    config = EvaluationConfig(
        enable_readability=True,
        enable_simplification=True,  # Enable to see before/after
        enable_semantic=False,
        enable_concept=False,
        enable_nli=False,
        enable_coverage=True,
        enable_ops=False,
        experiment_name="baseline-evals",
        dataset_name=f"baseline_{source_filter or 'all'}",
        model_version="reference",
    )
    
    orchestrator = EvaluationOrchestrator(config=config)
    
    results = orchestrator.evaluate_dataset(
        samples=eval_samples,
        run_name=f"baseline-n{len(eval_samples)}",
        tags={"eval_type": "baseline"},
    )
    
    print("\n" + "=" * 70)
    print("BASELINE METRICS (Reference Text Quality)")
    print("=" * 70)
    print(f"Samples: {results.sample_count}")
    
    if results.readability_comparison_mean:
        rc = results.readability_comparison_mean
        print(f"\n[READABILITY - Original vs Reference]")
        print(f"                           ORIGINAL  REFERENCE CHANGE")
        print(f"   FKGL (grade level):     {rc.original_fkgl:>5.1f}     {rc.simplified_fkgl:>5.1f}     {rc.fkgl_delta:+.1f}")
        print(f"   SMOG index:             {rc.original_smog:>5.1f}     {rc.simplified_smog:>5.1f}     {rc.smog_delta:+.1f}")
    
    if results.readability_mean:
        r = results.readability_mean
        print(f"\n[READABILITY - Target metrics for pipeline to match]")
        print(f"   FKGL (grade level):     {r.fkgl:.1f}")
        print(f"   SMOG index:             {r.smog:.1f}")
        print(f"   Avg sentence length:    {r.avg_sentence_length:.1f} words")
        print(f"   Jargon density:         {r.jargon_density:.2%}")
    
    if results.coverage_mean:
        cv = results.coverage_mean
        print(f"\n[COVERAGE]")
        print(f"   Reference CUI recall:   {cv.reference_cui_recall:.2%}")
        print(f"   Measurements preserved: {cv.key_measurements_present:.2%}")
    
    print("\n" + "=" * 70)
    
    if output_path:
        with open(output_path, "w") as f:
            json.dump(results.model_dump(mode="json"), f, indent=2)
        logger.info(f"Results saved to {output_path}")


@app.command()
def stats(
    dataset_path: Optional[Path] = typer.Option(
        None,
        "--dataset", "-d",
        help="Path to the CSV dataset file.",
    ),
):
    """Print statistics about the dataset without running evaluations."""
    path = dataset_path or DEFAULT_DATASET_PATH
    
    if not path.exists():
        logger.error(f"Dataset not found: {path}")
        raise typer.Exit(1)
    
    logger.info(f"Analyzing dataset: {path}")
    
    sources = {}
    splits = {}
    total = 0
    empty_original = 0
    empty_simplified = 0
    
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            total += 1
            
            src = row.get('source_dataset', 'unknown')
            sources[src] = sources.get(src, 0) + 1
            
            split = row.get('split', 'unknown')
            splits[split] = splits.get(split, 0) + 1
            
            if not row.get('original_text', '').strip():
                empty_original += 1
            if not row.get('plain_language_text', '').strip():
                empty_simplified += 1
    
    print(f"\nDataset: {path.name}")
    print(f"Total rows: {total:,}")
    print(f"Empty original: {empty_original:,}")
    print(f"Empty simplified: {empty_simplified:,}")
    print(f"Valid samples: {total - max(empty_original, empty_simplified):,}")
    
    print(f"\nBy Source:")
    for src, count in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"  {src}: {count:,} ({count/total:.1%})")
    
    print(f"\nBy Split:")
    for split, count in sorted(splits.items(), key=lambda x: -x[1]):
        print(f"  {split}: {count:,} ({count/total:.1%})")


def _print_summary(results: DatasetEvaluation):
    """Print a summary of evaluation results."""
    print("\n" + "=" * 70)
    print("PIPELINE EVALUATION RESULTS")
    print("=" * 70)
    print(f"Dataset: {results.dataset_name}")
    print(f"Model Version: {results.model_version}")
    print(f"Samples: {results.sample_count}")
    print(f"MLflow Run: {results.run_id}")
    print("-" * 70)
    
    # Before/After Comparison (most important for simplification)
    if results.readability_comparison_mean:
        rc = results.readability_comparison_mean
        print(f"\n[READABILITY - BEFORE vs AFTER SIMPLIFICATION]")
        print(f"                           BEFORE    AFTER     CHANGE")
        print(f"   FKGL (grade level):     {rc.original_fkgl:>5.1f}     {rc.simplified_fkgl:>5.1f}     {rc.fkgl_delta:+.1f}")
        print(f"   SMOG index:             {rc.original_smog:>5.1f}     {rc.simplified_smog:>5.1f}     {rc.smog_delta:+.1f}")
        print(f"   Avg sentence length:    {rc.original_avg_sentence_length:>5.1f}     {rc.simplified_avg_sentence_length:>5.1f}     {rc.sentence_length_delta:+.1f}")
        print(f"   Jargon density:         {rc.original_jargon_density:>5.2%}    {rc.simplified_jargon_density:>5.2%}    {rc.jargon_reduction:+.2%} reduction")
        
        # Interpretation
        if rc.fkgl_delta < -2:
            print(f"   --> GOOD: Reduced reading level by {abs(rc.fkgl_delta):.1f} grades")
        elif rc.fkgl_delta > 0:
            print(f"   --> WARNING: Reading level INCREASED")
    
    # Simplification Quality
    if results.simplification_quality_mean:
        sq = results.simplification_quality_mean
        print(f"\n[SIMPLIFICATION QUALITY]")
        print(f"   Overall score:          {sq.simplification_score:.1%}")
        print(f"   Concept recall:         {sq.concept_recall:.1%}")
        print(f"   Critical concept recall:{sq.critical_concept_recall:.1%}")
        print(f"   Measurement preserved:  {sq.measurement_preservation:.1%}")
        print(f"   Expansion ratio:        {sq.expansion_ratio:.2f}x")
        
        if sq.missing_diagnoses:
            print(f"   Missing diagnoses:      {', '.join(sq.missing_diagnoses[:3])}")
        if sq.missing_medications:
            print(f"   Missing medications:    {', '.join(sq.missing_medications[:3])}")
        if sq.missing_measurements:
            print(f"   Missing measurements:   {', '.join(sq.missing_measurements[:3])}")
    
    if results.readability_mean:
        r = results.readability_mean
        print(f"\n[READABILITY - Generated Output (Detailed)]")
        print(f"   FKGL (grade level):     {r.fkgl:.1f}")
        print(f"   SMOG index:             {r.smog:.1f}")
        print(f"   Avg sentence length:    {r.avg_sentence_length:.1f} words")
        print(f"   Jargon density:         {r.jargon_density:.2%}")
    
    if results.semantic_mean:
        s = results.semantic_mean
        print(f"\n[SEMANTIC SIMILARITY vs Reference]")
        print(f"   Precision:              {s.precision:.4f}")
        print(f"   Recall:                 {s.recall:.4f}")
        print(f"   F1:                     {s.f1:.4f}")
    
    if results.concept_mean:
        c = results.concept_mean
        print(f"\n[CONCEPT OVERLAP vs Reference]")
        print(f"   Precision:              {c.cui_precision:.4f}")
        print(f"   Recall:                 {c.cui_recall:.4f}")
        print(f"   F1:                     {c.cui_f1:.4f}")
    
    if results.nli_distribution:
        n = results.nli_distribution
        print(f"\n[NLI - Generated vs Original]")
        print(f"   Entailment rate:        {n.entailment_rate:.2%}")
        print(f"   Neutral rate:           {n.neutral_rate:.2%}")
        print(f"   Contradiction rate:     {n.contradiction_rate:.2%}")
    
    if results.coverage_mean:
        cv = results.coverage_mean
        print(f"\n[CONCEPT PRESERVATION - Original Info Retained]")
        print(f"   Original CUI recall:    {cv.reference_cui_recall:.2%}")
        print(f"   Measurements preserved: {cv.key_measurements_present:.2%}")
    
    if results.ops:
        o = results.ops
        print(f"\n[OPERATIONS]")
        print(f"   Latency P50:            {o.latency_p50_ms:.1f} ms")
        print(f"   Latency P95:            {o.latency_p95_ms:.1f} ms")
        print(f"   Latency Mean:           {o.latency_mean_ms:.1f} ms")
        print(f"   Tokens per report:      {o.cost_per_report_tokens:.0f}")
        print(f"   First-pass success:     {o.first_pass_success_rate:.2%}")
    
    print("\n" + "=" * 70)


def _print_stats_summary(all_stats: list[SampleStats]):
    """Print a summary of per-sample stats."""
    if not all_stats:
        return
    
    successful = [s for s in all_stats if not s.error]
    failed = [s for s in all_stats if s.error]
    
    print("\n" + "-" * 70)
    print("PER-SAMPLE STATS SUMMARY (for scatter plots)")
    print("-" * 70)
    print(f"Total samples: {len(all_stats)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        # FKGL distribution
        fkgl_deltas = [s.fkgl_delta for s in successful]
        print(f"\nFKGL Delta (lower = better simplification):")
        print(f"  Min: {min(fkgl_deltas):+.1f} | Max: {max(fkgl_deltas):+.1f} | Avg: {sum(fkgl_deltas)/len(fkgl_deltas):+.1f}")
        
        # Latency distribution
        latencies = [s.latency_ms for s in successful]
        print(f"\nLatency (ms):")
        print(f"  Min: {min(latencies):.0f} | Max: {max(latencies):.0f} | Avg: {sum(latencies)/len(latencies):.0f}")
        
        # First-pass success
        first_pass = sum(1 for s in successful if s.first_pass_success)
        print(f"\nFirst-pass success: {first_pass}/{len(successful)} ({first_pass/len(successful):.1%})")
        
        # Needs review
        needs_review = sum(1 for s in successful if s.needs_review)
        print(f"Needs review: {needs_review}/{len(successful)} ({needs_review/len(successful):.1%})")
    
    print("-" * 70)


if __name__ == "__main__":
    app()
