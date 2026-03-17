"""Utility modules for file handling, NLP, and other helpers."""

from .file_handler import read_report, write_json, write_json_model, write_text, safe_json_loads
from .pdf_parser import extract_text_from_pdf
from .nlp import (
    extract_medical_entities,
    extract_drug_names,
    extract_numbers_with_context,
    split_into_sentences,
    find_best_matching_sentence,
    compute_sentence_similarity,
    is_medical_term,
    get_entity_type,
    MedicalEntity,
    ExtractedNumber,
)

__all__ = [
    # File handling
    "read_report",
    "write_json",
    "write_json_model",
    "write_text",
    "safe_json_loads",
    "extract_text_from_pdf",
    # NLP utilities
    "extract_medical_entities",
    "extract_drug_names",
    "extract_numbers_with_context",
    "split_into_sentences",
    "find_best_matching_sentence",
    "compute_sentence_similarity",
    "is_medical_term",
    "get_entity_type",
    "MedicalEntity",
    "ExtractedNumber",
]
