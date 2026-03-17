#!/usr/bin/env python3
"""
CLI script for running evaluations on medical report simplification.

Usage:
    # Evaluate from a JSON file with samples
    python scripts/run_evals.py --input samples.json --dataset my-dataset
    
    # Evaluate from pipeline output
    python scripts/run_evals.py --input results.json --format pipeline
    
    # Run specific evaluators only
    python scripts/run_evals.py --input samples.json --evaluators readability,semantic,ops
    
    # Export stats CSV for scatter plots
    python scripts/run_evals.py --input samples.json --stats-csv stats.csv
    
    # Compare two model versions
    python scripts/run_evals.py --input v1.json --model-version v1
    python scripts/run_evals.py --input v2.json --model-version v2
"""

import csv
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import typer

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.evaluation_service import EvaluationOrchestrator, EvaluationConfig
from models.evaluation import DatasetEvaluation, SampleEvaluation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S'
)

# Reduce noise from third-party libraries
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

logger = logging.getLogger("run_evals")

app = typer.Typer(
    name="run-evals",
    help="Run evaluations on medical report simplification outputs.",
)


@dataclass
class EvalStats:
    """Per-sample evaluation statistics."""
    sample_id: str
    
    # Text stats
    original_chars: int = 0
    original_words: int = 0
    simplified_chars: int = 0
    simplified_words: int = 0
    
    # Readability
    original_fkgl: float = 0.0
    simplified_fkgl: float = 0.0
    fkgl_delta: float = 0.0
    original_smog: float = 0.0
    simplified_smog: float = 0.0
    smog_delta: float = 0.0
    
    # Quality
    concept_recall: float = 0.0
    critical_concept_recall: float = 0.0
    measurement_preservation: float = 0.0
    simplification_score: float = 0.0
    
    # Semantic (if available)
    bertscore_f1: float = 0.0
    
    # Coverage
    cui_recall: float = 0.0
    
    # Error
    error: str = ""


def load_samples_from_json(path: Path) -> list[dict]:
    """Load samples from a JSON file with standard format."""
    with open(path) as f:
        data = json.load(f)
    
    # Handle list of samples directly
    if isinstance(data, list):
        return data
    
    # Handle dict with samples key
    if "samples" in data:
        return data["samples"]
    
    raise ValueError(f"Unknown JSON format in {path}")


def load_samples_from_pipeline_output(path: Path) -> list[dict]:
    """Load samples from pipeline output format (simplified_report)."""
    with open(path) as f:
        data = json.load(f)
    
    if "simplified_report" not in data:
        raise ValueError(f"Expected 'simplified_report' key in {path}")
    
    samples = []
    for i, chunk in enumerate(data["simplified_report"]):
        samples.append({
            "sample_id": f"chunk_{i}",
            "original_text": chunk.get("original_chunk", ""),
            "simplified_text": chunk.get("simplified_chunk", ""),
            "reference_text": None,
        })
    
    return samples


def parse_evaluators(evaluators_str: str) -> set[str]:
    """Parse comma-separated evaluator names."""
    valid = {"readability", "semantic", "concept", "nli", "coverage", "ops"}
    requested = {e.strip().lower() for e in evaluators_str.split(",") if e.strip()}
    
    invalid = requested - valid
    if invalid:
        raise typer.BadParameter(f"Unknown evaluators: {invalid}. Valid: {valid}")
    
    return requested


def evaluate_single_sample(
    sample: dict,
    orchestrator: EvaluationOrchestrator,
) -> tuple[SampleEvaluation, EvalStats]:
    """Evaluate a single sample and collect stats."""
    sample_id = sample.get("sample_id", "unknown")
    original_text = sample.get("original_text", "")
    simplified_text = sample.get("simplified_text", "")
    reference_text = sample.get("reference_text")
    
    stats = EvalStats(
        sample_id=sample_id,
        original_chars=len(original_text),
        original_words=len(original_text.split()),
        simplified_chars=len(simplified_text),
        simplified_words=len(simplified_text.split()),
    )
    
    try:
        result = orchestrator.evaluate_sample(
            original_text=original_text,
            simplified_text=simplified_text,
            reference_text=reference_text,
            sample_id=sample_id,
        )
        
        # Extract stats from result
        if result.readability_comparison:
            rc = result.readability_comparison
            stats.original_fkgl = rc.original_fkgl
            stats.simplified_fkgl = rc.simplified_fkgl
            stats.fkgl_delta = rc.fkgl_delta
            stats.original_smog = rc.original_smog
            stats.simplified_smog = rc.simplified_smog
            stats.smog_delta = rc.smog_delta
        
        if result.simplification_quality:
            sq = result.simplification_quality
            stats.concept_recall = sq.concept_recall
            stats.critical_concept_recall = sq.critical_concept_recall
            stats.measurement_preservation = sq.measurement_preservation
            stats.simplification_score = sq.simplification_score
        
        if result.semantic:
            stats.bertscore_f1 = result.semantic.f1
        
        if result.coverage:
            stats.cui_recall = result.coverage.reference_cui_recall
        
        return result, stats
        
    except Exception as e:
        stats.error = str(e)
        # Return empty result
        return SampleEvaluation(
            sample_id=sample_id,
            original_text=original_text,
            simplified_text=simplified_text,
            reference_text=reference_text,
        ), stats


def evaluate_samples_parallel(
    samples: list[dict],
    orchestrator: EvaluationOrchestrator,
    max_workers: int = 4,
) -> tuple[list[SampleEvaluation], list[EvalStats]]:
    """Evaluate multiple samples in parallel."""
    results = []
    all_stats = []
    completed = 0
    total = len(samples)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_sample = {
            executor.submit(evaluate_single_sample, sample, orchestrator): sample
            for sample in samples
        }
        
        for future in as_completed(future_to_sample):
            completed += 1
            sample = future_to_sample[future]
            
            try:
                result, stats = future.result()
                results.append(result)
                all_stats.append(stats)
                
                if stats.error:
                    logger.error(f"[{completed}/{total}] {stats.sample_id}: {stats.error}")
                else:
                    logger.info(
                        f"[{completed}/{total}] {stats.sample_id}: "
                        f"FKGL {stats.original_fkgl:.1f}→{stats.simplified_fkgl:.1f} "
                        f"({stats.fkgl_delta:+.1f})"
                    )
            except Exception as e:
                logger.error(f"[{completed}/{total}] {sample.get('sample_id', 'unknown')}: {e}")
                all_stats.append(EvalStats(
                    sample_id=sample.get('sample_id', 'unknown'),
                    error=str(e),
                ))
    
    return results, all_stats


def export_stats_csv(stats: list[EvalStats], path: Path):
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


@app.command()
def evaluate(
    input_path: Path = typer.Option(
        ...,
        "--input", "-i",
        help="Path to input JSON file with samples or pipeline output.",
    ),
    format: str = typer.Option(
        "samples",
        "--format", "-f",
        help="Input format: 'samples' (list of sample dicts) or 'pipeline' (pipeline output).",
    ),
    dataset_name: str = typer.Option(
        "eval-dataset",
        "--dataset", "-d",
        help="Name of the dataset for MLflow logging.",
    ),
    model_version: str = typer.Option(
        "v1",
        "--model-version", "-v",
        help="Model/prompt version identifier.",
    ),
    evaluators: Optional[str] = typer.Option(
        None,
        "--evaluators", "-e",
        help="Comma-separated list of evaluators to run (default: all).",
    ),
    parallel: int = typer.Option(
        1,
        "--parallel", "-p",
        help="Number of parallel workers (1 = sequential).",
    ),
    semantic_model: str = typer.Option(
        "pubmedbert",
        "--semantic-model",
        help="Semantic similarity model: 'pubmedbert' or 'bioclinicalbert'.",
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
        "medical-simplification-evals",
        "--experiment",
        help="MLflow experiment name.",
    ),
    tracking_uri: Optional[str] = typer.Option(
        None,
        "--tracking-uri",
        help="MLflow tracking server URI.",
    ),
    run_name: Optional[str] = typer.Option(
        None,
        "--run-name",
        help="Custom name for the MLflow run.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-V",
        help="Enable verbose logging.",
    ),
):
    """
    Run evaluations on a dataset of simplified medical reports.
    
    Computes metrics across samples and logs results to MLflow.
    Supports parallel processing and stats export for analysis.
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info(f"Loading samples from {input_path}")
    
    # Load samples
    if format == "samples":
        samples = load_samples_from_json(input_path)
    elif format == "pipeline":
        samples = load_samples_from_pipeline_output(input_path)
    else:
        raise typer.BadParameter(f"Unknown format: {format}")
    
    logger.info(f"Loaded {len(samples)} samples")
    
    # Configure evaluators
    enabled_evaluators = None
    if evaluators:
        enabled_evaluators = parse_evaluators(evaluators)
    
    config = EvaluationConfig(
        enable_readability=(enabled_evaluators is None or "readability" in enabled_evaluators),
        enable_simplification=True,  # Always enable for before/after stats
        enable_semantic=(enabled_evaluators is None or "semantic" in enabled_evaluators),
        enable_concept=(enabled_evaluators is None or "concept" in enabled_evaluators),
        enable_nli=(enabled_evaluators is None or "nli" in enabled_evaluators),
        enable_coverage=(enabled_evaluators is None or "coverage" in enabled_evaluators),
        enable_ops=(enabled_evaluators is None or "ops" in enabled_evaluators),
        semantic_model=semantic_model,
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
        dataset_name=dataset_name,
        model_version=model_version,
    )
    
    # Create orchestrator
    logger.info("Initializing evaluation orchestrator...")
    orchestrator = EvaluationOrchestrator(config=config)
    
    # Run evaluations
    logger.info(f"Running evaluations on {len(samples)} samples...")
    if parallel > 1:
        logger.info(f"Using {parallel} parallel workers")
        sample_results, all_stats = evaluate_samples_parallel(
            samples, orchestrator, max_workers=parallel
        )
    else:
        # Sequential
        sample_results = []
        all_stats = []
        for i, sample in enumerate(samples):
            logger.info(f"[{i+1}/{len(samples)}] Evaluating {sample.get('sample_id', i)}...")
            result, stats = evaluate_single_sample(sample, orchestrator)
            sample_results.append(result)
            all_stats.append(stats)
    
    # Export stats CSV if requested
    if stats_csv:
        export_stats_csv(all_stats, stats_csv)
    
    # Aggregate into dataset result
    results = orchestrator._aggregate_results(sample_results, run_name or "manual-run")
    results.dataset_name = dataset_name
    results.model_version = model_version
    
    # Print summary
    _print_summary(results)
    
    # Print stats summary
    _print_stats_summary(all_stats)
    
    # Save results if output path provided
    if output_path:
        with open(output_path, "w") as f:
            json.dump(results.model_dump(mode="json"), f, indent=2)
        logger.info(f"Results saved to {output_path}")
    
    logger.info(f"MLflow run ID: {results.run_id}")
    logger.info("Evaluation complete!")


def _print_summary(results: DatasetEvaluation):
    """Print a summary of evaluation results."""
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Dataset: {results.dataset_name}")
    print(f"Model Version: {results.model_version}")
    print(f"Samples: {results.sample_count}")
    print(f"MLflow Run: {results.run_id}")
    print("-" * 70)
    
    if results.readability_comparison_mean:
        rc = results.readability_comparison_mean
        print(f"\n[READABILITY - BEFORE vs AFTER]")
        print(f"                           BEFORE    AFTER     CHANGE")
        print(f"   FKGL (grade level):     {rc.original_fkgl:>5.1f}     {rc.simplified_fkgl:>5.1f}     {rc.fkgl_delta:+.1f}")
        print(f"   SMOG index:             {rc.original_smog:>5.1f}     {rc.simplified_smog:>5.1f}     {rc.smog_delta:+.1f}")
    
    if results.simplification_quality_mean:
        sq = results.simplification_quality_mean
        print(f"\n[SIMPLIFICATION QUALITY]")
        print(f"   Overall score:          {sq.simplification_score:.1%}")
        print(f"   Concept recall:         {sq.concept_recall:.1%}")
        print(f"   Critical recall:        {sq.critical_concept_recall:.1%}")
        print(f"   Measurement preserved:  {sq.measurement_preservation:.1%}")
    
    if results.readability_mean:
        r = results.readability_mean
        print(f"\n[READABILITY - Output]")
        print(f"   FKGL (grade level):     {r.fkgl:.1f}")
        print(f"   SMOG index:             {r.smog:.1f}")
        print(f"   Avg sentence length:    {r.avg_sentence_length:.1f} words")
        print(f"   Jargon density:         {r.jargon_density:.2%}")
    
    if results.semantic_mean:
        s = results.semantic_mean
        print(f"\n[SEMANTIC SIMILARITY (BERTScore)]")
        print(f"   Precision:              {s.precision:.4f}")
        print(f"   Recall:                 {s.recall:.4f}")
        print(f"   F1:                     {s.f1:.4f}")
    
    if results.concept_mean:
        c = results.concept_mean
        print(f"\n[CONCEPT OVERLAP (UMLS CUIs)]")
        print(f"   Precision:              {c.cui_precision:.4f}")
        print(f"   Recall:                 {c.cui_recall:.4f}")
        print(f"   F1:                     {c.cui_f1:.4f}")
    
    if results.nli_distribution:
        n = results.nli_distribution
        print(f"\n[NLI DISTRIBUTION]")
        print(f"   Entailment rate:        {n.entailment_rate:.2%}")
        print(f"   Neutral rate:           {n.neutral_rate:.2%}")
        print(f"   Contradiction rate:     {n.contradiction_rate:.2%}")
    
    if results.coverage_mean:
        cv = results.coverage_mean
        print(f"\n[COVERAGE]")
        print(f"   CUI recall:             {cv.reference_cui_recall:.2%}")
        print(f"   Measurements preserved: {cv.key_measurements_present:.2%}")
    
    if results.ops:
        o = results.ops
        print(f"\n[OPERATIONS]")
        print(f"   Latency P50:            {o.latency_p50_ms:.1f} ms")
        print(f"   Latency P95:            {o.latency_p95_ms:.1f} ms")
        print(f"   Tokens per report:      {o.cost_per_report_tokens:.0f}")
        print(f"   First-pass success:     {o.first_pass_success_rate:.2%}")
    
    print("\n" + "=" * 70)


def _print_stats_summary(all_stats: list[EvalStats]):
    """Print summary of per-sample stats."""
    if not all_stats:
        return
    
    successful = [s for s in all_stats if not s.error]
    
    if not successful:
        return
    
    print("\n" + "-" * 70)
    print("PER-SAMPLE STATS (for scatter plots)")
    print("-" * 70)
    
    # FKGL distribution
    fkgl_deltas = [s.fkgl_delta for s in successful]
    print(f"FKGL Delta: Min={min(fkgl_deltas):+.1f} Max={max(fkgl_deltas):+.1f} Avg={sum(fkgl_deltas)/len(fkgl_deltas):+.1f}")
    
    # Simplification score distribution
    scores = [s.simplification_score for s in successful if s.simplification_score > 0]
    if scores:
        print(f"Simplification Score: Min={min(scores):.1%} Max={max(scores):.1%} Avg={sum(scores)/len(scores):.1%}")
    
    print("-" * 70)


@app.command()
def quick_eval(
    original: str = typer.Argument(..., help="Original text or path to file"),
    simplified: str = typer.Argument(..., help="Simplified text or path to file"),
    reference: Optional[str] = typer.Argument(None, help="Reference text (optional)"),
):
    """
    Quick evaluation of a single original/simplified pair.
    
    Can pass text directly or paths to files.
    """
    # Handle file paths
    def resolve_text(text: str) -> str:
        path = Path(text)
        if path.exists():
            return path.read_text()
        return text
    
    original_text = resolve_text(original)
    simplified_text = resolve_text(simplified)
    reference_text = resolve_text(reference) if reference else None
    
    # Create orchestrator with fast config
    config = EvaluationConfig(
        enable_simplification=True,
        enable_nli=False,   # Skip NLI for speed
    )
    orchestrator = EvaluationOrchestrator(config=config)
    
    # Evaluate
    result = orchestrator.evaluate_sample(
        original_text=original_text,
        simplified_text=simplified_text,
        reference_text=reference_text,
        sample_id="quick_eval",
    )
    
    # Print results
    print("\n" + "=" * 50)
    print("QUICK EVALUATION RESULTS")
    print("=" * 50)
    
    if result.readability_comparison:
        rc = result.readability_comparison
        print(f"\nReadability Change:")
        print(f"  FKGL: {rc.original_fkgl:.1f} → {rc.simplified_fkgl:.1f} ({rc.fkgl_delta:+.1f})")
        print(f"  SMOG: {rc.original_smog:.1f} → {rc.simplified_smog:.1f} ({rc.smog_delta:+.1f})")
    
    if result.simplification_quality:
        sq = result.simplification_quality
        print(f"\nSimplification Quality:")
        print(f"  Score: {sq.simplification_score:.1%}")
        print(f"  Concept recall: {sq.concept_recall:.1%}")
        print(f"  Critical recall: {sq.critical_concept_recall:.1%}")
        print(f"  Measurements: {sq.measurement_preservation:.1%}")
    
    if result.readability:
        r = result.readability
        print(f"\nOutput Readability:")
        print(f"  FKGL: {r.fkgl:.1f} | SMOG: {r.smog:.1f} | Jargon: {r.jargon_density:.2%}")
    
    if result.semantic:
        s = result.semantic
        print(f"\nSemantic (BERTScore): P={s.precision:.3f} R={s.recall:.3f} F1={s.f1:.3f}")
    
    if result.coverage:
        cv = result.coverage
        print(f"\nCoverage: CUI recall={cv.reference_cui_recall:.2%} Measurements={cv.key_measurements_present:.2%}")
    
    print("\n" + "=" * 50)


if __name__ == "__main__":
    app()
