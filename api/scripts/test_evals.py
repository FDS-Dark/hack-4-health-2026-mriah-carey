#!/usr/bin/env python3
"""
Test script for the evaluation framework.

Run this to verify evaluators are working correctly:
    python scripts/test_evals.py
    
Examples:
    # Test all evaluators
    python scripts/test_evals.py --mode all
    
    # Test specific evaluator
    python scripts/test_evals.py --mode readability
    
    # Test with custom sample
    python scripts/test_evals.py --mode quick --original "CT scan shows..." --simplified "The scan shows..."
"""

import json
import logging
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S'
)

# Reduce noise
for name in ["urllib3", "httpx", "transformers", "sentence_transformers", "filelock"]:
    logging.getLogger(name).setLevel(logging.WARNING)

logger = logging.getLogger("test_evals")

# Sample medical texts for testing
SAMPLE_ORIGINAL = """
RADIOLOGY REPORT

EXAM: CT Chest with Contrast
DATE: 01/20/2026
PATIENT: John Doe
DOB: 01/15/1965

CLINICAL HISTORY:
60-year-old male with history of smoking, presenting with persistent cough and shortness of breath.

TECHNIQUE:
CT of the chest was performed with IV contrast. Images were obtained from the lung apices through the adrenal glands.

FINDINGS:

LUNGS AND AIRWAYS:
There is a 4 mm ground-glass nodule in the right upper lobe (series 3, image 45).
No solid pulmonary nodules identified.
The airways are patent to the segmental level bilaterally.
No consolidation or pleural effusion.

MEDIASTINUM AND HILA:
The heart size is normal.
No pericardial effusion.
No mediastinal or hilar lymphadenopathy.
The thoracic aorta is normal in caliber.

IMPRESSION:
1. 4 mm ground-glass nodule in the right upper lobe. Given patient's smoking history, recommend follow-up CT in 12 months per Fleischner Society guidelines.
2. No acute cardiopulmonary abnormality.
"""

SAMPLE_SIMPLIFIED = """
CT SCAN RESULTS - CHEST

Patient: John Doe (born 01/15/1965)
Test Date: 01/20/2026

Why this test was done:
You are a 60-year-old man with a history of smoking. You came in because of a cough that won't go away and trouble breathing.

What the pictures show:

Your Lungs:
- There is a tiny spot (4 mm, about the size of a grain of rice) in your right lung. This spot looks like "ground glass" on the scan.
- No other spots or lumps were found in your lungs.
- Your airways are open and working normally.
- There is no fluid around your lungs.

Your Heart Area:
- Your heart looks normal in size.
- No fluid around your heart.
- No swollen lymph nodes in your chest.
- The main blood vessel from your heart looks normal.

What this means:
1. The small 4 mm spot in your right lung needs to be watched. Because you have a smoking history, your doctor recommends another CT scan in 12 months to make sure it doesn't grow.
2. Everything else looks healthy - no signs of infection or other problems.

IMPORTANT: Please talk to your doctor about these results and any questions you have.
"""


def test_readability():
    """Test the readability evaluator."""
    from evaluators.readability import ReadabilityEvaluator
    
    print("\n" + "=" * 60)
    print("TESTING: ReadabilityEvaluator")
    print("=" * 60)
    
    evaluator = ReadabilityEvaluator()
    
    # Evaluate simplified text
    result = evaluator.evaluate_sample(
        original_text=SAMPLE_ORIGINAL,
        simplified_text=SAMPLE_SIMPLIFIED,
    )
    
    print(f"FKGL (grade level): {result.fkgl}")
    print(f"SMOG index: {result.smog}")
    print(f"Avg sentence length: {result.avg_sentence_length} words")
    print(f"Jargon density: {result.jargon_density:.2%}")
    print(f"Total tokens: {result.total_tokens}")
    print(f"CUI count: {result.cui_count}")
    
    # Sanity checks
    assert result.fkgl > 0, "FKGL should be positive"
    assert result.smog > 0, "SMOG should be positive"
    assert result.avg_sentence_length > 0, "Avg sentence length should be positive"
    
    print("\n[OK] ReadabilityEvaluator test PASSED")
    return result


def test_semantic():
    """Test the semantic similarity evaluator."""
    from evaluators.semantic import SemanticSimilarityEvaluator
    
    print("\n" + "=" * 60)
    print("TESTING: SemanticSimilarityEvaluator")
    print("=" * 60)
    
    evaluator = SemanticSimilarityEvaluator(model_name="pubmedbert")
    
    result = evaluator.evaluate_sample(
        original_text=SAMPLE_ORIGINAL,
        simplified_text=SAMPLE_SIMPLIFIED,
    )
    
    print(f"Precision: {result.precision}")
    print(f"Recall: {result.recall}")
    print(f"F1: {result.f1}")
    print(f"Model: {result.model_name}")
    
    # Sanity checks
    assert 0 <= result.precision <= 1, "Precision should be in [0, 1]"
    assert 0 <= result.recall <= 1, "Recall should be in [0, 1]"
    assert 0 <= result.f1 <= 1, "F1 should be in [0, 1]"
    
    print("\n[OK] SemanticSimilarityEvaluator test PASSED")
    return result


def test_nli():
    """Test the NLI distribution evaluator."""
    from evaluators.nli import NLIDistributionEvaluator
    
    print("\n" + "=" * 60)
    print("TESTING: NLIDistributionEvaluator")
    print("=" * 60)
    
    evaluator = NLIDistributionEvaluator()
    
    result = evaluator.evaluate_sample(
        original_text=SAMPLE_ORIGINAL,
        simplified_text=SAMPLE_SIMPLIFIED,
    )
    
    print(f"Entailment rate: {result.entailment_rate:.2%}")
    print(f"Neutral rate: {result.neutral_rate:.2%}")
    print(f"Contradiction rate: {result.contradiction_rate:.2%}")
    print(f"Total pairs: {result.total_pairs}")
    
    # Sanity checks
    total_rate = result.entailment_rate + result.neutral_rate + result.contradiction_rate
    assert abs(total_rate - 1.0) < 0.01 or result.total_pairs == 0, "Rates should sum to 1"
    
    print("\n[OK] NLIDistributionEvaluator test PASSED")
    return result


def test_coverage():
    """Test the coverage evaluator."""
    from evaluators.coverage import CoverageEvaluator
    
    print("\n" + "=" * 60)
    print("TESTING: CoverageEvaluator")
    print("=" * 60)
    
    evaluator = CoverageEvaluator()
    
    result = evaluator.evaluate_sample(
        original_text=SAMPLE_ORIGINAL,
        simplified_text=SAMPLE_SIMPLIFIED,
    )
    
    print(f"Reference CUI recall: {result.reference_cui_recall:.2%}")
    print(f"Measurements preserved: {result.key_measurements_present:.2%}")
    print(f"Total reference CUIs: {result.total_reference_cuis}")
    print(f"Total measurements: {result.total_measurements}")
    print(f"Preserved measurements: {result.preserved_measurements}")
    
    print("\n[OK] CoverageEvaluator test PASSED")
    return result


def test_ops():
    """Test the ops evaluator."""
    from evaluators.ops import OpsEvaluator, PipelineTimer
    
    print("\n" + "=" * 60)
    print("TESTING: OpsEvaluator")
    print("=" * 60)
    
    evaluator = OpsEvaluator()
    
    # Simulate some pipeline runs
    import time
    for i in range(5):
        with PipelineTimer(evaluator) as timer:
            time.sleep(0.01 * (i + 1))  # Varying latencies
            timer.input_tokens = 100 * (i + 1)
            timer.output_tokens = 50 * (i + 1)
            timer.first_pass_success = (i % 2 == 0)
            timer.refinement_count = 1 if i % 3 == 0 else 0
    
    result = evaluator.get_metrics()
    
    print(f"Latency P50: {result.latency_p50_ms:.1f} ms")
    print(f"Latency P95: {result.latency_p95_ms:.1f} ms")
    print(f"Latency mean: {result.latency_mean_ms:.1f} ms")
    print(f"Tokens per report: {result.cost_per_report_tokens:.0f}")
    print(f"First-pass success rate: {result.first_pass_success_rate:.2%}")
    print(f"Refinement rate: {result.refinement_rate:.2%}")
    
    print("\n[OK] OpsEvaluator test PASSED")
    return result


def test_orchestrator():
    """Test the full evaluation orchestrator."""
    from services.evaluation_service import EvaluationOrchestrator, EvaluationConfig
    
    print("\n" + "=" * 60)
    print("TESTING: EvaluationOrchestrator")
    print("=" * 60)
    
    # Create config with fast evaluators only
    config = EvaluationConfig(
        enable_readability=True,
        enable_semantic=False,  # Skip semantic for speed
        enable_concept=False,  # Skip concept (needs OpenSearch)
        enable_nli=False,  # Skip NLI for speed
        enable_coverage=True,
        enable_ops=True,
        dataset_name="test-dataset",
        model_version="test-v1",
    )
    
    orchestrator = EvaluationOrchestrator(config=config)
    
    # Evaluate single sample
    result = orchestrator.evaluate_sample(
        original_text=SAMPLE_ORIGINAL,
        simplified_text=SAMPLE_SIMPLIFIED,
        sample_id="test_sample",
    )
    
    print(f"Sample ID: {result.sample_id}")
    print(f"Has readability: {result.readability is not None}")
    print(f"Has coverage: {result.coverage is not None}")
    
    if result.readability:
        print(f"  Readability FKGL: {result.readability.fkgl}")
    
    if result.coverage:
        print(f"  Coverage measurements: {result.coverage.key_measurements_present:.2%}")
    
    print("\n[OK] EvaluationOrchestrator test PASSED")
    return result


def main():
    """Run all tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test evaluation framework")
    parser.add_argument(
        "--mode",
        choices=["all", "readability", "semantic", "nli", "coverage", "ops", "orchestrator", "quick"],
        default="all",
        help="Which test(s) to run",
    )
    parser.add_argument("--original", type=str, help="Original text for quick mode")
    parser.add_argument("--simplified", type=str, help="Simplified text for quick mode")
    
    args = parser.parse_args()
    
    if args.mode == "quick":
        if not args.original or not args.simplified:
            print("Error: --original and --simplified required for quick mode")
            sys.exit(1)
        
        from evaluators.readability import ReadabilityEvaluator
        evaluator = ReadabilityEvaluator()
        result = evaluator.evaluate_sample(
            original_text=args.original,
            simplified_text=args.simplified,
        )
        print(json.dumps(result.model_dump(), indent=2))
        return
    
    tests = {
        "readability": test_readability,
        "semantic": test_semantic,
        "nli": test_nli,
        "coverage": test_coverage,
        "ops": test_ops,
        "orchestrator": test_orchestrator,
    }
    
    if args.mode == "all":
        # Run tests in order of increasing complexity/time
        order = ["readability", "coverage", "ops", "orchestrator"]
        logger.info(f"Running {len(order)} fast tests (skipping slow model tests)...")
        
        for name in order:
            tests[name]()
        
        print("\n" + "=" * 60)
        print("ALL FAST TESTS PASSED!")
        print("=" * 60)
        print("\nTo run slow model tests:")
        print("  python scripts/test_evals.py --mode semantic")
        print("  python scripts/test_evals.py --mode nli")
    else:
        tests[args.mode]()


if __name__ == "__main__":
    main()
