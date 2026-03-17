"""
NLP utilities for medical text processing.

Uses scispacy/medspacy for proper biomedical entity extraction
instead of hardcoded keyword lists.
"""

import logging
import re
from dataclasses import dataclass
from functools import lru_cache

logger = logging.getLogger("utils.nlp")

# Lazy-loaded models
_nlp_model = None
_sentence_model = None


@dataclass
class MedicalEntity:
    """A medical entity extracted from text."""
    text: str
    label: str  # Entity type (DRUG, DISEASE, PROCEDURE, etc.)
    start: int
    end: int
    
    def __hash__(self):
        return hash((self.text.lower(), self.label))
    
    def __eq__(self, other):
        if not isinstance(other, MedicalEntity):
            return False
        return self.text.lower() == other.text.lower() and self.label == other.label


@dataclass
class ExtractedNumber:
    """A number extracted with context."""
    value: float
    text: str  # Original text representation
    unit: str | None
    context: str  # Surrounding text


def get_nlp_model():
    """
    Get the scispacy NLP model (lazy loaded).
    
    Uses en_core_sci_lg for biomedical entity recognition.
    """
    global _nlp_model
    
    if _nlp_model is not None:
        return _nlp_model
    
    logger.info("Loading scispacy model...")
    
    try:
        import spacy
        
        # Try to load the large biomedical model
        try:
            _nlp_model = spacy.load("en_core_sci_lg")
            logger.info("Loaded en_core_sci_lg model")
        except OSError:
            # Fall back to smaller model
            try:
                _nlp_model = spacy.load("en_core_sci_sm")
                logger.info("Loaded en_core_sci_sm model (fallback)")
            except OSError:
                # Last resort: use basic English model
                _nlp_model = spacy.load("en_core_web_sm")
                logger.warning("Using en_core_web_sm - install scispacy models for better medical NER")
        
        return _nlp_model
        
    except Exception as e:
        logger.error(f"Failed to load spacy model: {e}")
        return None


def get_sentence_model():
    """Get the sentence transformer model for embeddings (lazy loaded)."""
    global _sentence_model
    
    if _sentence_model is not None:
        return _sentence_model
    
    logger.info("Loading sentence transformer model...")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Use BioClinicalBERT for medical text
        _sentence_model = SentenceTransformer("emilyalsentzer/Bio_ClinicalBERT")
        logger.info("Loaded Bio_ClinicalBERT model")
        return _sentence_model
        
    except Exception as e:
        logger.error(f"Failed to load sentence model: {e}")
        return None


def extract_medical_entities(text: str) -> list[MedicalEntity]:
    """
    Extract medical entities from text using scispacy.
    
    Returns entities with labels like:
    - DRUG (medications)
    - DISEASE (conditions, diagnoses)
    - PROCEDURE (medical procedures)
    - ANATOMY (body parts)
    - CHEMICAL (chemicals, compounds)
    """
    nlp = get_nlp_model()
    if nlp is None:
        return []
    
    # Process text
    doc = nlp(text)
    
    entities = []
    for ent in doc.ents:
        # Map scispacy labels to simpler categories
        label = _map_entity_label(ent.label_)
        
        entities.append(MedicalEntity(
            text=ent.text,
            label=label,
            start=ent.start_char,
            end=ent.end_char
        ))
    
    return entities


def _map_entity_label(label: str) -> str:
    """Map scispacy entity labels to simplified categories."""
    # scispacy uses UMLS semantic types
    # Map to simpler categories for our use case
    
    drug_labels = {"CHEMICAL", "DRUG"}
    disease_labels = {"DISEASE", "DISORDER", "FINDING"}
    procedure_labels = {"PROCEDURE", "TREATMENT"}
    anatomy_labels = {"ANATOMY", "BODY_PART", "ORGAN"}
    
    label_upper = label.upper()
    
    if label_upper in drug_labels or "DRUG" in label_upper or "CHEM" in label_upper:
        return "DRUG"
    elif label_upper in disease_labels or "DISEASE" in label_upper:
        return "DISEASE"
    elif label_upper in procedure_labels or "PROCEDURE" in label_upper:
        return "PROCEDURE"
    elif label_upper in anatomy_labels or "ANAT" in label_upper:
        return "ANATOMY"
    else:
        return label_upper


def extract_drug_names(text: str) -> set[str]:
    """Extract medication/drug names from text."""
    entities = extract_medical_entities(text)
    drugs = {e.text.lower() for e in entities if e.label == "DRUG"}
    
    # Also use regex patterns for common drug name patterns
    # (brand names often end in specific suffixes)
    drug_patterns = [
        r'\b\w+(?:mab|nib|zole|mycin|cillin|pril|sartan|statin|olol|dipine|pam|lam|zepam)\b',
        r'\b(?:aspirin|ibuprofen|acetaminophen|tylenol|advil|motrin)\b',
    ]
    
    for pattern in drug_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            drugs.add(match.group(0).lower())
    
    return drugs


def extract_numbers_with_context(text: str) -> list[ExtractedNumber]:
    """
    Extract numbers with their surrounding context.
    
    Uses NLP to understand what the numbers refer to.
    """
    results = []
    
    # Pattern for numbers with optional units
    number_pattern = re.compile(
        r'(\d+(?:\.\d+)?)\s*'
        r'(mm|cm|m|kg|g|mg|mcg|μg|ml|mL|L|cc|'
        r'mmHg|bpm|%|°[CF]|IU|units?|mEq|mmol|'
        r'mg/dL|g/dL|kg/m²|lb|oz|inches?|in|")?',
        re.IGNORECASE
    )
    
    for match in number_pattern.finditer(text):
        value_str = match.group(1)
        unit = match.group(2)
        
        try:
            value = float(value_str)
        except ValueError:
            continue
        
        # Get context (30 chars before and after)
        start = max(0, match.start() - 30)
        end = min(len(text), match.end() + 30)
        context = text[start:end]
        
        results.append(ExtractedNumber(
            value=value,
            text=value_str,
            unit=unit.lower() if unit else None,
            context=context
        ))
    
    return results


def split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences using NLP.
    
    Better than regex for handling abbreviations, decimals, etc.
    """
    nlp = get_nlp_model()
    
    if nlp is not None:
        # Use spacy's sentence segmentation
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    else:
        # Fallback to regex
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
    
    # Also split on bullet points and newlines
    expanded = []
    for sent in sentences:
        parts = re.split(r'[\n•*]+', sent)
        for part in parts:
            cleaned = part.strip()
            if cleaned and len(cleaned) > 10:
                expanded.append(cleaned)
    
    return expanded


def compute_sentence_similarity(sent1: str, sent2: str) -> float:
    """
    Compute semantic similarity between two sentences.
    
    Uses sentence embeddings for semantic comparison.
    """
    model = get_sentence_model()
    
    if model is None:
        # Fallback to word overlap
        words1 = set(sent1.lower().split())
        words2 = set(sent2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union)
    
    from sklearn.metrics.pairwise import cosine_similarity
    
    embeddings = model.encode([sent1, sent2])
    sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    
    return float(sim)


def find_best_matching_sentence(
    query: str,
    candidates: list[str],
    min_similarity: float = 0.3
) -> tuple[str | None, float]:
    """
    Find the candidate sentence that best matches the query.
    
    Uses a combination of:
    1. Medical entity overlap (drugs, diseases, etc.)
    2. Number overlap
    3. Semantic similarity
    
    Returns (best_match, score) or (None, 0) if no good match.
    """
    if not candidates:
        return None, 0.0
    
    # Extract entities and numbers from query
    query_entities = {e.text.lower() for e in extract_medical_entities(query)}
    query_numbers = {n.value for n in extract_numbers_with_context(query)}
    query_drugs = extract_drug_names(query)
    
    best_match = None
    best_score = 0.0
    
    for candidate in candidates:
        # Extract entities and numbers from candidate
        cand_entities = {e.text.lower() for e in extract_medical_entities(candidate)}
        cand_numbers = {n.value for n in extract_numbers_with_context(candidate)}
        cand_drugs = extract_drug_names(candidate)
        
        # Calculate overlap scores
        entity_overlap = _jaccard_similarity(query_entities, cand_entities)
        number_overlap = _jaccard_similarity(query_numbers, cand_numbers)
        drug_overlap = _jaccard_similarity(query_drugs, cand_drugs)
        
        # Word overlap as baseline
        query_words = set(query.lower().split())
        cand_words = set(candidate.lower().split())
        word_overlap = _jaccard_similarity(query_words, cand_words)
        
        # Weighted score:
        # - Drug overlap is most important (if both mention drugs, they should match)
        # - Number overlap is next (measurements should match)
        # - Entity overlap next
        # - Word overlap as baseline
        score = (
            0.35 * drug_overlap +
            0.25 * number_overlap +
            0.25 * entity_overlap +
            0.15 * word_overlap
        )
        
        if score > best_score:
            best_score = score
            best_match = candidate
    
    # Only return if score meets minimum threshold
    if best_score >= min_similarity:
        return best_match, best_score
    
    return None, 0.0


def _jaccard_similarity(set1: set, set2: set) -> float:
    """Calculate Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 0.0
    if not set1 or not set2:
        return 0.0
    
    intersection = set1 & set2
    union = set1 | set2
    
    return len(intersection) / len(union) if union else 0.0


def is_medical_term(text: str) -> bool:
    """
    Check if a term is a medical term using NLP.
    
    Returns True if the term is recognized as a medical entity.
    """
    entities = extract_medical_entities(text)
    return len(entities) > 0


@lru_cache(maxsize=1000)
def get_entity_type(term: str) -> str | None:
    """
    Get the entity type for a term (cached).
    
    Returns the entity type (DRUG, DISEASE, etc.) or None.
    """
    entities = extract_medical_entities(term)
    if entities:
        return entities[0].label
    return None
