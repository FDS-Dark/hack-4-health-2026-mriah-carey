"""V5: Recommendation policy validator using NLI entailment checks."""

import logging
import re
from dataclasses import dataclass
from enum import Enum

from models.validation import (
    ErrorCode,
    Severity,
    ValidationError,
    ValidationResult,
)
from validators.base import BaseValidator

logger = logging.getLogger("validators.recommendation_policy")


class ActionCategory(str, Enum):
    """Categories of actionable medical statements."""
    EMERGENCY_DIRECTIVE = "emergency_directive"
    MEDICATION_ADVICE = "medication_advice"
    DIAGNOSIS_CLAIM = "diagnosis_claim"
    URGENCY_CLAIM = "urgency_claim"
    PROGNOSIS_CLAIM = "prognosis_claim"
    TREATMENT_RECOMMENDATION = "treatment_recommendation"


@dataclass
class ActionSentence:
    """A detected action sentence with its category."""
    sentence: str
    category: ActionCategory
    severity: Severity
    matched_phrase: str


@dataclass
class NLIResult:
    """Result of NLI inference."""
    label: str  # "entailment", "neutral", "contradiction"
    confidence: float


class RecommendationPolicyValidator(BaseValidator):
    """
    Validates that simplified text doesn't contain actionable medical advice
    unless it's entailed by the original report.
    
    Uses NLI to check if action sentences in the summary are supported by
    the original text, reducing false positives from phrasing differences.
    """
    
    name = "recommendation_policy_validator"
    
    # Patterns to detect action sentences
    ACTION_DETECTION_PATTERNS = [
        # Emergency directives
        (re.compile(r'[^.]*\b(go to|visit|call|contact|seek)\b[^.]*\b(er|emergency|hospital|doctor|911|care|attention|help)\b[^.]*', re.IGNORECASE),
         ActionCategory.EMERGENCY_DIRECTIVE, Severity.CRITICAL),
        
        # Medication advice
        (re.compile(r'[^.]*\b(start|begin|take|stop|discontinue|prescribe)\b[^.]*\b(medication|medicine|drug|prescription|treatment|dose)\b[^.]*', re.IGNORECASE),
         ActionCategory.MEDICATION_ADVICE, Severity.CRITICAL),
        (re.compile(r'[^.]*\byou should\s+(start|begin|take|stop|continue)\b[^.]*', re.IGNORECASE),
         ActionCategory.MEDICATION_ADVICE, Severity.HIGH),
        
        # Diagnosis claims
        (re.compile(r'[^.]*\b(this is|you have|diagnosed with|confirms?|consistent with)\b[^.]*\b(cancer|malignant|malignancy|tumor|carcinoma)\b[^.]*', re.IGNORECASE),
         ActionCategory.DIAGNOSIS_CLAIM, Severity.CRITICAL),
        (re.compile(r'[^.]*\b(this is|you have|confirms?)\b[^.]*\b(stroke|heart attack|infarction|embolism|aneurysm|thrombosis)\b[^.]*', re.IGNORECASE),
         ActionCategory.DIAGNOSIS_CLAIM, Severity.CRITICAL),
        
        # Urgency claims
        (re.compile(r'[^.]*\b(urgent|urgently|emergency|immediately|life.threatening|critical condition)\b[^.]*', re.IGNORECASE),
         ActionCategory.URGENCY_CLAIM, Severity.HIGH),
        (re.compile(r'[^.]*\brequires?\s+(immediate|urgent|emergency)\b[^.]*', re.IGNORECASE),
         ActionCategory.URGENCY_CLAIM, Severity.HIGH),
        
        # Prognosis claims
        (re.compile(r'[^.]*\b(will|going to)\s+(die|survive|recover|spread|metastasize|worsen)\b[^.]*', re.IGNORECASE),
         ActionCategory.PROGNOSIS_CLAIM, Severity.CRITICAL),
        (re.compile(r'[^.]*\b(terminal|incurable|fatal|deadly|hopeless)\b[^.]*', re.IGNORECASE),
         ActionCategory.PROGNOSIS_CLAIM, Severity.CRITICAL),
        
        # Treatment recommendations
        (re.compile(r'[^.]*\b(need|needs|require|requires|must have)\b[^.]*\b(surgery|chemotherapy|radiation|treatment|biopsy|procedure)\b[^.]*', re.IGNORECASE),
         ActionCategory.TREATMENT_RECOMMENDATION, Severity.HIGH),
        (re.compile(r'[^.]*\bshould\s+(have|get|undergo|schedule|receive)\b[^.]*\b(surgery|chemotherapy|radiation|treatment|biopsy|procedure|scan)\b[^.]*', re.IGNORECASE),
         ActionCategory.TREATMENT_RECOMMENDATION, Severity.HIGH),
        (re.compile(r'[^.]*\brecommend(s|ed|ation)?\b[^.]*\b(follow.?up|biopsy|surgery|treatment|evaluation|consultation)\b[^.]*', re.IGNORECASE),
         ActionCategory.TREATMENT_RECOMMENDATION, Severity.MEDIUM),
    ]
    
    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        entailment_threshold: float = 0.5,
        use_similarity_fallback: bool = True,
    ):
        self.model_name = model_name
        self.entailment_threshold = entailment_threshold
        self.use_similarity_fallback = use_similarity_fallback
        self._nli_model = None
        self._nli_tokenizer = None
        self._similarity_model = None
    
    def _load_nli_model(self) -> bool:
        """Lazy load the NLI model."""
        if self._nli_model is not None:
            return True
        
        logger.info(f"Loading NLI model: {self.model_name}")
        
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            
            self._nli_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._nli_model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            logger.info("NLI model loaded successfully")
            return True
        except Exception as e:
            logger.warning(f"Failed to load NLI model: {e}")
            
            if self.use_similarity_fallback:
                try:
                    from sentence_transformers import SentenceTransformer
                    self._similarity_model = SentenceTransformer("emilyalsentzer/Bio_ClinicalBERT")
                    logger.info("Loaded fallback similarity model")
                    return True
                except Exception as e2:
                    logger.error(f"Failed to load fallback model: {e2}")
            return False
    
    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    def _detect_action_sentences(self, text: str) -> list[ActionSentence]:
        """Detect action sentences in text using pattern matching."""
        actions = []
        seen_sentences = set()
        
        for pattern, category, severity in self.ACTION_DETECTION_PATTERNS:
            for match in pattern.finditer(text):
                matched_text = match.group(0).strip()
                
                sentence_start = text.rfind('.', 0, match.start())
                sentence_start = sentence_start + 1 if sentence_start != -1 else 0
                sentence_end = text.find('.', match.end())
                sentence_end = sentence_end + 1 if sentence_end != -1 else len(text)
                
                sentence = text[sentence_start:sentence_end].strip()
                
                if sentence in seen_sentences:
                    continue
                seen_sentences.add(sentence)
                
                actions.append(ActionSentence(
                    sentence=sentence,
                    category=category,
                    severity=severity,
                    matched_phrase=matched_text,
                ))
        
        return actions
    
    def _find_best_matching_sentence(self, query: str, candidates: list[str]) -> str | None:
        """Find the sentence in candidates that best matches the query."""
        if not candidates:
            return None
        
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been", 
                      "have", "has", "had", "do", "does", "did", "will", "would",
                      "could", "should", "may", "might", "must", "to", "of", "in",
                      "for", "on", "with", "at", "by", "from", "as", "this", "that",
                      "it", "its", "or", "and", "but", "if", "so", "you", "your"}
        query_words -= stop_words
        
        best_match = None
        best_score = 0.0
        
        for candidate in candidates:
            candidate_words = set(re.findall(r'\b\w+\b', candidate.lower()))
            candidate_words -= stop_words
            
            if not candidate_words or not query_words:
                continue
            
            overlap = len(query_words & candidate_words)
            union = len(query_words | candidate_words)
            score = overlap / union if union > 0 else 0
            
            if score > best_score:
                best_score = score
                best_match = candidate
        
        return best_match
    
    def _run_nli(self, premise: str, hypothesis: str) -> NLIResult:
        """Run NLI inference."""
        if not self._load_nli_model():
            return NLIResult(label="neutral", confidence=0.5)
        
        if self._nli_model is not None and self._nli_tokenizer is not None:
            import torch
            
            inputs = self._nli_tokenizer(
                premise, hypothesis,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            
            with torch.no_grad():
                outputs = self._nli_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)[0]
            
            labels = ["contradiction", "neutral", "entailment"]
            max_idx = probs.argmax().item()
            
            result = NLIResult(
                label=labels[max_idx],
                confidence=probs[max_idx].item()
            )
            
            logger.debug(f"NLI: '{hypothesis[:40]}...' → {result.label} ({result.confidence:.2%})")
            return result
        
        # Fallback: similarity
        if self._similarity_model is not None:
            from sklearn.metrics.pairwise import cosine_similarity
            
            embeddings = self._similarity_model.encode([premise, hypothesis])
            sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            
            if sim > 0.75:
                result = NLIResult(label="entailment", confidence=sim)
            elif sim > 0.5:
                result = NLIResult(label="neutral", confidence=0.5)
            else:
                result = NLIResult(label="contradiction", confidence=1.0 - sim)
            
            logger.debug(f"Similarity: {sim:.2f} → {result.label}")
            return result
        
        return NLIResult(label="neutral", confidence=0.5)
    
    def _check_entailment(
        self,
        action: ActionSentence,
        original_sentences: list[str],
        original_text: str,
    ) -> tuple[bool, NLIResult, str | None]:
        """Check if an action sentence is entailed by the original text."""
        best_premise = self._find_best_matching_sentence(action.sentence, original_sentences)
        
        if best_premise is None:
            best_premise = original_text[:1000]
        
        result = self._run_nli(premise=best_premise, hypothesis=action.sentence)
        
        is_entailed = (
            result.label == "entailment" and 
            result.confidence >= self.entailment_threshold
        )
        
        return is_entailed, result, best_premise
    
    def validate(self, original_text: str, simplified_text: str) -> ValidationResult:
        """Validate that action sentences are entailed by original."""
        logger.info(f"{'='*60}")
        logger.info(f"RECOMMENDATION POLICY VALIDATOR - Starting validation")
        logger.info(f"{'='*60}")
        
        errors: list[ValidationError] = []
        
        # Detect action sentences
        action_sentences = self._detect_action_sentences(simplified_text)
        logger.info(f"Detected {len(action_sentences)} action sentences")
        
        for i, action in enumerate(action_sentences):
            logger.debug(f"  Action {i+1} [{action.category.value}]: '{action.matched_phrase}'")
        
        if not action_sentences:
            logger.info("No action sentences detected - PASSED ✓")
            return ValidationResult.success(self.name)
        
        # Get original sentences
        original_sentences = self._split_sentences(original_text)
        logger.debug(f"Original has {len(original_sentences)} sentences")
        
        # Check each action via NLI
        entailed_count = 0
        failed_count = 0
        
        for action in action_sentences:
            is_entailed, nli_result, premise = self._check_entailment(
                action=action,
                original_sentences=original_sentences,
                original_text=original_text,
            )
            
            if is_entailed:
                entailed_count += 1
                logger.info(f"  ✓ ENTAILED: '{action.matched_phrase[:50]}...' ({nli_result.confidence:.2%})")
            else:
                failed_count += 1
                logger.warning(f"  ✗ NOT ENTAILED: '{action.matched_phrase[:50]}...'")
                logger.warning(f"    Category: {action.category.value}")
                logger.warning(f"    NLI: {nli_result.label} ({nli_result.confidence:.2%})")
                if premise:
                    logger.warning(f"    Best premise: '{premise[:60]}...'")
                
                if nli_result.label == "contradiction":
                    message = f"Action sentence contradicts original (NLI: {nli_result.label}, conf: {nli_result.confidence:.2%})"
                else:
                    message = f"Action sentence not supported by original (NLI: {nli_result.label}, conf: {nli_result.confidence:.2%})"
                
                errors.append(ValidationError(
                    code=ErrorCode.UNSAFE_RECOMMENDATION,
                    severity=action.severity,
                    message=message,
                    summary_span=action.sentence[:150] + "..." if len(action.sentence) > 150 else action.sentence,
                    source_evidence=premise[:150] + "..." if premise and len(premise) > 150 else premise,
                    metadata={
                        "category": action.category.value,
                        "nli_label": nli_result.label,
                        "nli_confidence": nli_result.confidence,
                        "matched_phrase": action.matched_phrase,
                    },
                    fix=f"Rephrase or remove '{action.matched_phrase}'"
                ))
        
        logger.info(f"Results: {entailed_count} entailed, {failed_count} failed")
        
        if errors:
            logger.warning(f"RECOMMENDATION POLICY VALIDATOR - FAILED with {len(errors)} errors")
            return ValidationResult.failure(self.name, errors)
        
        logger.info("RECOMMENDATION POLICY VALIDATOR - PASSED ✓")
        return ValidationResult.success(self.name)
