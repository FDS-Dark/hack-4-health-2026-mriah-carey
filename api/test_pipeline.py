#!/usr/bin/env python3
"""
Test script for the medical report processing pipeline.

Run this directly to test the pipeline without using the API:
    python test_pipeline.py

Examples:
    # Test validators with sample data
    python test_pipeline.py --mode validators
    
    # Test with a text file
    python test_pipeline.py --file medical_report.txt --mode validators
    
    # Test with a PDF file
    python test_pipeline.py --file report.pdf --mode validators
    
    # Run full pipeline (requires Gemini API key)
    python test_pipeline.py --file report.pdf --mode full
    
    # Test individual validators
    python test_pipeline.py --mode individual
"""

import json
import logging
import sys
from pathlib import Path

# Configure logging to see debug output
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S'
)

# Reduce noise from third-party libraries
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("opensearchpy").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("medspacy").setLevel(logging.WARNING)
logging.getLogger("spacy").setLevel(logging.WARNING)
logging.getLogger("PyRuSH").setLevel(logging.WARNING)
logging.getLogger("google_genai").setLevel(logging.WARNING)
logging.getLogger("google_genai.models").setLevel(logging.WARNING)


# Filter to suppress medspacy regex syntax warnings
class MedspaCyWarningFilter(logging.Filter):
    """Filter out medspacy regex syntax warnings."""
    def filter(self, record):
        # Suppress the "not a eligible syntax" warnings from medspacy
        if "eligible syntax" in record.getMessage():
            return False
        return True


# Apply filter to root logger
logging.getLogger().addFilter(MedspaCyWarningFilter())

logger = logging.getLogger("test_pipeline")


# Sample medical report for testing
SAMPLE_REPORT = """
RADIOLOGY REPORT

Patient: John Doe
DOB: 01/15/1965
Date of Exam: 01/20/2026
Exam: CT Chest with Contrast

CLINICAL HISTORY:
60-year-old male with history of smoking, presenting with persistent cough and shortness of breath. 
Rule out pulmonary nodule or mass.

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

PLEURA:
No pleural thickening or effusion bilaterally.

CHEST WALL AND AXILLA:
No suspicious axillary lymphadenopathy.
Degenerative changes of the thoracic spine.

UPPER ABDOMEN:
Limited evaluation of the upper abdomen shows no focal hepatic lesion.
Adrenal glands are unremarkable.

IMPRESSION:
1. 4 mm ground-glass nodule in the right upper lobe. Given patient's smoking history, 
   recommend follow-up CT in 12 months per Fleischner Society guidelines.
2. No evidence of mediastinal or hilar lymphadenopathy.
3. No pleural effusion.
4. Degenerative changes of the thoracic spine.

Electronically signed by:
Dr. Jane Smith, MD
Radiologist
"""

# Sample simplified text with some issues for testing validators
SAMPLE_SIMPLIFIED = """
Your CT scan of your chest showed a small spot in your right lung. This spot is 4 mm in size and looks like ground glass. 

The doctors did not find any other spots or lumps in your lungs. Your airways look clear. Your heart looks normal. There is no fluid around your heart or lungs.

Your spine shows some wear and tear, which is normal for your age. Your liver and other organs in your upper belly look fine.

**What this means:**
- You have a small 4 mm ground-glass spot in your right upper lung
- You need a follow-up CT scan in 12 months
- This is being watched because of your smoking history
- No signs of cancer spreading to lymph nodes
- Your spine has normal aging changes

Please talk to your doctor about the follow-up scan.
"""


def load_report_from_file(file_path: str) -> str:
    """
    Load a medical report from a file.
    
    Supports:
    - .txt files (UTF-8 or Latin-1 encoding)
    - .pdf files (uses pypdf for extraction)
    
    Args:
        file_path: Path to the file
        
    Returns:
        Extracted text content
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    logger.info(f"Loading file: {path}")
    logger.info(f"File type: {path.suffix}")
    logger.info(f"File size: {path.stat().st_size:,} bytes")
    
    if path.suffix.lower() == ".pdf":
        logger.info("Parsing PDF file...")
        from utils.pdf_parser import extract_text_from_pdf
        
        with open(path, "rb") as f:
            pdf_bytes = f.read()
        
        text = extract_text_from_pdf(pdf_bytes)
        logger.info(f"Extracted {len(text):,} characters from PDF")
        return text
    
    elif path.suffix.lower() in (".txt", ".text", ".md"):
        logger.info("Reading text file...")
        
        # Try UTF-8 first, fall back to Latin-1
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            logger.warning("UTF-8 decode failed, trying Latin-1...")
            text = path.read_text(encoding="latin-1")
        
        logger.info(f"Read {len(text):,} characters from text file")
        return text
    
    else:
        # Try to read as text anyway
        logger.warning(f"Unknown file extension '{path.suffix}', attempting to read as text...")
        try:
            text = path.read_text(encoding="utf-8")
            logger.info(f"Read {len(text):,} characters")
            return text
        except Exception as e:
            raise ValueError(f"Cannot read file {path}: {e}")


def test_validators_only(original_text: str, simplified_text: str):
    """Test just the validators without running the full pipeline."""
    from validators.orchestrator import create_default_orchestrator, create_fast_orchestrator
    
    logger.info("\n" + "="*80)
    logger.info("TESTING VALIDATORS ONLY")
    logger.info("="*80 + "\n")
    
    logger.info(f"Original text: {len(original_text):,} characters")
    logger.info(f"Simplified text: {len(simplified_text):,} characters")
    
    # Use fast orchestrator to avoid model loading delays
    logger.info("\nCreating fast orchestrator (numeric + recommendation policy only)...")
    orchestrator = create_fast_orchestrator()
    
    result = orchestrator.validate(original_text, simplified_text)
    
    logger.info("\n" + "-"*80)
    logger.info("VALIDATION RESULT")
    logger.info("-"*80)
    logger.info(f"Passed: {result.passed}")
    logger.info(f"Needs review: {result.needs_review}")
    logger.info(f"Total errors: {len(result.all_errors)}")
    
    if result.all_errors:
        logger.info("\nErrors:")
        for i, err in enumerate(result.all_errors, 1):
            logger.info(f"  {i}. [{err.code.value}] {err.message}")
            if err.summary_span:
                logger.info(f"     Summary: {err.summary_span[:80]}...")
            if err.fix:
                logger.info(f"     Fix: {err.fix}")
    
    # Print JSON result
    print("\n" + "="*80)
    print("VALIDATION RESULT (JSON)")
    print("="*80)
    print(json.dumps(result.model_dump(), indent=2, default=str))
    
    return result


def test_full_validators(original_text: str, simplified_text: str):
    """Test all validators including NLI (slower, loads models)."""
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from validators.orchestrator import create_default_orchestrator
    from validators.numeric import NumericValidator
    from validators.umls_grounding import UMLSGroundingValidator
    from validators.contradiction_nli import ContradictionNLIValidator
    from validators.recommendation_policy import RecommendationPolicyValidator
    from validators.empty_chunk import EmptyChunkValidator
    
    logger.info("\n" + "="*80)
    logger.info("TESTING ALL VALIDATORS (PARALLEL)")
    logger.info("="*80 + "\n")
    
    logger.info(f"Original text: {len(original_text):,} characters")
    logger.info(f"Simplified text: {len(simplified_text):,} characters")
    logger.info("\nThis will load NLI models - may take a moment on first run...")
    
    # Create individual validators for parallel testing
    validators = [
        ("empty_chunk", EmptyChunkValidator()),
        ("numeric", NumericValidator()),
        ("umls_grounding", UMLSGroundingValidator()),
        ("contradiction_nli", ContradictionNLIValidator()),
        ("recommendation_policy", RecommendationPolicyValidator()),
    ]
    
    start_time = time.time()
    results = {}
    
    # Run validators in parallel
    with ThreadPoolExecutor(max_workers=len(validators)) as executor:
        future_to_name = {
            executor.submit(v.validate, original_text, simplified_text): name
            for name, v in validators
        }
        
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                result = future.result()
                results[name] = result
                status = "✓" if result.passed else f"✗ ({len(result.errors)} errors)"
                logger.info(f"  {name}: {status}")
            except Exception as e:
                logger.error(f"  {name}: FAILED with exception: {e}")
    
    elapsed = time.time() - start_time
    
    logger.info("\n" + "-"*80)
    logger.info("VALIDATION SUMMARY")
    logger.info("-"*80)
    logger.info(f"Total time: {elapsed:.2f} seconds")
    
    all_passed = all(r.passed for r in results.values())
    total_errors = sum(len(r.errors) for r in results.values())
    
    logger.info(f"All passed: {all_passed}")
    logger.info(f"Total errors: {total_errors}")
    
    # Print JSON result
    print("\n" + "="*80)
    print("VALIDATION RESULTS (JSON)")
    print("="*80)
    print(json.dumps({name: r.model_dump() for name, r in results.items()}, indent=2, default=str))
    
    return results


def test_full_pipeline(report_text: str, parallel: bool = True):
    """Test the full pipeline with validation and repair loop."""
    import time
    from pipeline import process_medical_report, PipelineConfig
    
    mode_str = "PARALLEL" if parallel else "SEQUENTIAL"
    logger.info("\n" + "="*80)
    logger.info(f"TESTING FULL PIPELINE ({mode_str} MODE)")
    logger.info("="*80 + "\n")
    
    # Configure pipeline
    config = PipelineConfig(
        max_repair_iterations=2,
        enable_legacy_comparison=True,
        enable_validation=True,
        enable_exa_sources=False,  # Disable Exa for testing
        parallel_chunks=parallel,
        max_parallel_workers=4,
    )
    
    logger.info(f"Pipeline config: max_repair_iterations={config.max_repair_iterations}")
    logger.info(f"Pipeline config: parallel_chunks={config.parallel_chunks}")
    logger.info(f"Pipeline config: max_parallel_workers={config.max_parallel_workers}")
    logger.info(f"Report length: {len(report_text):,} characters")
    logger.info("\nStarting pipeline...\n")
    
    try:
        start_time = time.time()
        result = process_medical_report(report_text, config=config)
        elapsed_time = time.time() - start_time
        
        logger.info("\n" + "="*80)
        logger.info("PIPELINE RESULTS")
        logger.info("="*80)
        logger.info(f"Total time: {elapsed_time:.2f} seconds")
        logger.info(f"Chunks processed: {len(result['simplified_report'])}")
        logger.info(f"Avg time per chunk: {elapsed_time / len(result['simplified_report']):.2f} seconds")
        
        for i, chunk in enumerate(result["simplified_report"], 1):
            logger.info(f"\n--- Chunk {i} ---")
            logger.info(f"Original ({len(chunk['original_chunk']):,} chars):")
            logger.info(f"  {chunk['original_chunk'][:100]}...")
            logger.info(f"\nSimplified ({len(chunk['simplified_chunk']):,} chars):")
            logger.info(f"  {chunk['simplified_chunk'][:100]}...")
            logger.info(f"\nSimilarity: {chunk['similarity']:.4f}")
            logger.info(f"Needs review: {chunk.get('needs_review', False)}")
            
            if chunk.get('validation'):
                validation = chunk['validation']
                logger.info(f"Validation passed: {validation['passed']}")
                if validation.get('results'):
                    for vr in validation['results']:
                        status = "✓" if vr['passed'] else "✗"
                        logger.info(f"  {status} {vr['validator_name']}: {len(vr.get('errors', []))} errors")
        
        # Print JSON result
        print("\n" + "="*80)
        print("PIPELINE RESULT (JSON)")
        print("="*80)
        print(json.dumps(result, indent=2, default=str))
        
        return result
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


def test_individual_validators(original_text: str, simplified_text: str):
    """Test each validator individually with detailed output."""
    from validators.numeric import NumericValidator
    from validators.recommendation_policy import RecommendationPolicyValidator
    from validators.evidence_span import EvidenceSpanValidator
    
    logger.info("\n" + "="*80)
    logger.info("TESTING INDIVIDUAL VALIDATORS")
    logger.info("="*80 + "\n")
    
    logger.info(f"Original text: {len(original_text):,} characters")
    logger.info(f"Simplified text: {len(simplified_text):,} characters")
    
    validators_to_test = [
        ("NumericValidator", NumericValidator()),
        ("EvidenceSpanValidator", EvidenceSpanValidator()),
        ("RecommendationPolicyValidator", RecommendationPolicyValidator()),
    ]
    
    results = {}
    
    for name, validator in validators_to_test:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing {name}...")
        logger.info(f"{'='*60}")
        
        result = validator.validate(original_text, simplified_text)
        results[name] = result
        
        print(f"\n{name} result:")
        print(json.dumps(result.model_dump(), indent=2, default=str))
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    
    for name, result in results.items():
        status = "✓ PASSED" if result.passed else f"✗ FAILED ({len(result.errors)} errors)"
        logger.info(f"  {name}: {status}")
    
    return results


def test_pdf_parsing(file_path: str):
    """Test just the PDF parsing without any validation."""
    logger.info("\n" + "="*80)
    logger.info("TESTING PDF PARSING ONLY")
    logger.info("="*80 + "\n")
    
    text = load_report_from_file(file_path)
    
    logger.info(f"\nExtracted text ({len(text):,} characters):")
    logger.info("-"*60)
    
    # Print first 2000 characters
    if len(text) > 2000:
        print(text[:2000])
        print(f"\n... ({len(text) - 2000:,} more characters)")
    else:
        print(text)
    
    return text


def main():
    """Main entry point for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test the medical report pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_pipeline.py --mode validators
      Test validators with sample data (fast)
      
  python test_pipeline.py --file report.pdf --mode validators
      Load PDF and test validators
      
  python test_pipeline.py --file report.txt --mode full
      Run full pipeline with text file
      
  python test_pipeline.py --mode full-validators
      Run all validators including NLI models
      
  python test_pipeline.py --file report.pdf --mode parse-only
      Just test PDF parsing
"""
    )
    
    parser.add_argument(
        "--file", "-f", 
        help="Path to a medical report file (.txt or .pdf)"
    )
    parser.add_argument(
        "--mode", "-m", 
        choices=["full", "full-sequential", "validators", "full-validators", "individual", "parse-only"], 
        default="validators",
        help="Test mode: full (parallel), full-sequential, validators (fast), full-validators (with NLI), individual, parse-only"
    )
    parser.add_argument(
        "--parallel", "-p",
        action="store_true",
        default=True,
        help="Enable parallel chunk processing (default: True)"
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel chunk processing"
    )
    parser.add_argument(
        "--simplified", "-s", 
        help="Custom simplified text to validate against (for validators mode)"
    )
    parser.add_argument(
        "--log-level", "-l",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="DEBUG",
        help="Logging level"
    )
    parser.add_argument(
        "--output", "-o",
        help="Save results to file (supports .json, .txt, .html)"
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory to save output files (default: output)"
    )
    
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Handle parse-only mode
    if args.mode == "parse-only":
        if not args.file:
            logger.error("--file is required for parse-only mode")
            sys.exit(1)
        test_pdf_parsing(args.file)
        return
    
    # Load report
    if args.file:
        logger.info(f"Loading report from: {args.file}")
        try:
            report_text = load_report_from_file(args.file)
        except Exception as e:
            logger.error(f"Failed to load file: {e}")
            sys.exit(1)
    else:
        logger.info("Using sample report (use --file to load a custom file)")
        report_text = SAMPLE_REPORT
    
    logger.info(f"Report loaded: {len(report_text):,} characters")
    
    # Get simplified text
    if args.simplified:
        simplified_text = args.simplified
    else:
        simplified_text = SAMPLE_SIMPLIFIED
    
    # Determine parallel mode
    parallel = args.parallel and not args.no_parallel
    
    # Run tests based on mode
    result = None
    if args.mode == "full":
        result = test_full_pipeline(report_text, parallel=parallel)
    elif args.mode == "full-sequential":
        result = test_full_pipeline(report_text, parallel=False)
    elif args.mode == "full-validators":
        result = test_full_validators(report_text, simplified_text)
    elif args.mode == "individual":
        result = test_individual_validators(report_text, simplified_text)
    else:  # validators (fast)
        result = test_validators_only(report_text, simplified_text)
    
    # Save results if --output specified
    if args.output and result:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert result to JSON-serializable format
        if hasattr(result, 'model_dump'):
            output_data = result.model_dump()
        elif isinstance(result, dict):
            # Handle dict of results (from individual/full-validators)
            output_data = {k: v.model_dump() if hasattr(v, 'model_dump') else v for k, v in result.items()}
        else:
            output_data = result
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        logger.info(f"\nResults saved to: {output_path}")
    
    logger.info("\n" + "="*80)
    logger.info("TEST COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()
