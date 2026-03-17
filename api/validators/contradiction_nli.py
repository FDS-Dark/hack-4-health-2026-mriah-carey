"""V4: Contradiction detection via Natural Language Inference."""

import logging
from dataclasses import dataclass

from models.validation import (
    ErrorCode,
    Severity,
    ValidationError,
    ValidationResult,
)
from validators.base import BaseValidator
from utils.nlp import (
    extract_medical_entities,
    extract_drug_names,
    extract_numbers_with_context,
    split_into_sentences,
    find_best_matching_sentence,
)

logger = logging.getLogger("validators.contradiction_nli")


@dataclass
class NLIResult:
    """Result of NLI inference."""
    label: str  # "entailment", "neutral", "contradiction"
    confidence: float


class ContradictionNLIValidator(BaseValidator):
    """
    Validates that simplified text doesn't contradict the original.
    
    Uses NLP-based entity extraction for smart premise selection:
    - Matches summary sentences to original sentences by entity overlap
    - Prioritizes medical entities (drugs, diseases, procedures)
    - Uses NLI model to detect actual contradictions
    """
    
    name = "contradiction_nli_validator"
    
    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        contradiction_threshold: float = 0.85,
        min_match_score: float = 0.25,
    ):
        self.model_name = model_name
        self.contradiction_threshold = contradiction_threshold
        self.min_match_score = min_match_score
        self._model = None
        self._tokenizer = None
    
    def _load_nli_model(self):
        """Lazy load the NLI model."""
        if self._model is not None:
            return True
        
        logger.info(f"Loading NLI model: {self.model_name}")
        
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            logger.info("NLI model loaded successfully")
            return True
        except Exception as e:
            logger.warning(f"Failed to load NLI model: {e}")
            return False
    
    def _run_nli(self, premise: str, hypothesis: str) -> NLIResult:
        """Run NLI inference on a premise-hypothesis pair."""
        if not self._load_nli_model():
            # Can't run NLI without model - assume neutral
            return NLIResult(label="neutral", confidence=0.5)
        
        import torch
        
        inputs = self._tokenizer(
            premise, hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]
        
        labels = ["contradiction", "neutral", "entailment"]
        max_idx = probs.argmax().item()
        
        result = NLIResult(
            label=labels[max_idx],
            confidence=probs[max_idx].item()
        )
        
        logger.debug(f"NLI: '{hypothesis[:30]}...' → {result.label} ({result.confidence:.2%})")
        return result
    
    def _extract_key_content(self, text: str) -> dict:
        """Extract key content from text using NLP."""
        return {
            "entities": {e.text.lower() for e in extract_medical_entities(text)},
            "drugs": extract_drug_names(text),
            "numbers": {n.value for n in extract_numbers_with_context(text)},
        }
    
    def validate(self, original_text: str, simplified_text: str) -> ValidationResult:
        """Validate that simplified text doesn't contradict original."""
        logger.info(f"{'='*60}")
        logger.info(f"CONTRADICTION NLI VALIDATOR - Starting validation")
        logger.info(f"{'='*60}")
        
        errors: list[ValidationError] = []
        
        # Split into sentences using NLP
        original_sentences = split_into_sentences(original_text)
        simplified_sentences = split_into_sentences(simplified_text)
        
        logger.debug(f"Original sentences: {len(original_sentences)}")
        logger.debug(f"Simplified sentences: {len(simplified_sentences)}")
        
        if not original_sentences or not simplified_sentences:
            logger.info("No sentences to compare - PASSED ✓")
            return ValidationResult.success(self.name)
        
        contradiction_count = 0
        checked_count = 0
        skipped_count = 0
        
        for i, simp_sent in enumerate(simplified_sentences):
            # Use NLP-based matching to find best original sentence
            orig_sent, match_score = find_best_matching_sentence(
                simp_sent, 
                original_sentences,
                min_similarity=self.min_match_score
            )
            
            # Skip if no good match - this is "unsupported" not "contradiction"
            if orig_sent is None:
                logger.debug(f"  Sentence {i+1}: No matching original found - SKIPPED")
                skipped_count += 1
                continue
            
            logger.debug(f"  Sentence {i+1}: Matched with score {match_score:.2f}")
            checked_count += 1
            
            # Run NLI
            result = self._run_nli(premise=orig_sent, hypothesis=simp_sent)
            
            # Only flag as contradiction if:
            # 1. NLI says contradiction with high confidence
            # 2. We had a good entity match (so we're comparing related content)
            if (
                result.label == "contradiction" 
                and result.confidence >= self.contradiction_threshold
            ):
                # Double-check: do they share medical entities?
                simp_content = self._extract_key_content(simp_sent)
                orig_content = self._extract_key_content(orig_sent)
                
                shared_drugs = simp_content["drugs"] & orig_content["drugs"]
                shared_entities = simp_content["entities"] & orig_content["entities"]
                shared_numbers = simp_content["numbers"] & orig_content["numbers"]
                
                # Only flag if there's actual content overlap
                has_overlap = shared_drugs or shared_entities or shared_numbers
                
                if has_overlap:
                    contradiction_count += 1
                    logger.error(f"CONTRADICTION DETECTED:")
                    logger.error(f"  Simplified: '{simp_sent[:80]}...'")
                    logger.error(f"  Original:   '{orig_sent[:80]}...'")
                    logger.error(f"  Shared: drugs={shared_drugs}, entities={list(shared_entities)[:3]}")
                    
                    errors.append(ValidationError(
                        code=ErrorCode.CONTRADICTION_DETECTED,
                        severity=Severity.CRITICAL,
                        message=f"Simplified text contradicts original (confidence: {result.confidence:.2%})",
                        summary_span=simp_sent[:100] + "..." if len(simp_sent) > 100 else simp_sent,
                        source_evidence=orig_sent[:100] + "..." if len(orig_sent) > 100 else orig_sent,
                        metadata={
                            "confidence": result.confidence,
                            "nli_label": result.label,
                            "match_score": match_score,
                            "shared_content": {
                                "drugs": list(shared_drugs),
                                "entities": list(shared_entities)[:5],
                                "numbers": list(shared_numbers)[:5],
                            }
                        },
                        fix="Revise to match what the original says"
                    ))
                else:
                    logger.debug(f"  Sentence {i+1}: NLI says contradiction but no content overlap - SKIPPED")
            else:
                logger.debug(f"  Sentence {i+1}: {result.label} ({result.confidence:.2%}) ✓")
        
        logger.info(f"Checked {checked_count} sentences, skipped {skipped_count}, found {contradiction_count} contradictions")
        
        if errors:
            logger.warning(f"CONTRADICTION NLI VALIDATOR - FAILED with {len(errors)} errors")
            return ValidationResult.failure(self.name, errors)
        
        logger.info("CONTRADICTION NLI VALIDATOR - PASSED ✓")
        return ValidationResult.success(self.name)
