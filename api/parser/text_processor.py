"""Text preprocessing utilities for medical reports."""

import re
from typing import List, Optional


def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Raw text input
        
    Returns:
        Cleaned text with normalized whitespace
    """
    # Normalize whitespace (multiple spaces/newlines to single)
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


def detect_sections(text: str) -> Optional[List[tuple]]:
    """
    Detect common medical report sections.
    
    Args:
        text: Medical report text
        
    Returns:
        List of (section_name, start_index, end_index) tuples, or None if no sections found
    """
    # Common medical report section headers (case-insensitive)
    section_patterns = [
        r'^(CHIEF COMPLAINT|CC):',
        r'^(HISTORY OF PRESENT ILLNESS|HPI):',
        r'^(PAST MEDICAL HISTORY|PMH):',
        r'^(PHYSICAL EXAMINATION|PE|PHYSICAL EXAM):',
        r'^(ASSESSMENT|ASSESSMENT AND PLAN|A&P):',
        r'^(PLAN):',
        r'^(DIAGNOSIS):',
        r'^(MEDICATIONS):',
        r'^(LABORATORY|LABS):',
        r'^(IMAGING):',
    ]
    
    sections = []
    lines = text.split('\n')
    
    for i, line in enumerate(lines):
        line_upper = line.strip().upper()
        for pattern in section_patterns:
            if re.match(pattern, line_upper):
                section_name = line.strip()
                # Find the start of this section's content (next non-empty line)
                start_idx = i
                sections.append((section_name, start_idx, None))
                break
    
    # If sections found, calculate end indices
    if sections:
        for i in range(len(sections)):
            start_idx = sections[i][1]
            # End is either start of next section or end of document
            if i + 1 < len(sections):
                end_idx = sections[i + 1][1]
            else:
                end_idx = len(lines)
            sections[i] = (sections[i][0], start_idx, end_idx)
        return sections
    
    return None


def preprocess_report(text: str) -> str:
    """
    Preprocess a medical report for extraction.
    
    Args:
        text: Raw medical report text
        
    Returns:
        Preprocessed text ready for extraction
    """
    # Clean the text
    cleaned = clean_text(text)
    
    # Additional preprocessing can be added here
    # (e.g., remove headers/footers, normalize dates, etc.)
    
    return cleaned
