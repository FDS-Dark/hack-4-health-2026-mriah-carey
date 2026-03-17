"""Service for simplifying medical text using Gemini."""

import json
import logging

from clients.gemini import GeminiClient, get_gemini_client
from services.concept_extractor import extract_concepts_for_prompt

logger = logging.getLogger("services.simplifier")


class SimplifierService:
    """Service for simplifying medical text to patient-friendly language."""
    
    def __init__(self, gemini_client: GeminiClient | None = None, opensearch_client=None):
        """
        Initialize the simplifier service.
        
        Args:
            gemini_client: Optional Gemini client. Uses default if not provided.
            opensearch_client: Optional OpenSearch client for UMLS concept extraction.
        """
        self.client = gemini_client or get_gemini_client()
        self.opensearch_client = opensearch_client
    
    def _format_sources(self, sources: str | None) -> str:
        """Format sources for inclusion in prompts."""
        if not sources:
            return "No sources available."
        
        try:
            sources_data = json.loads(sources) if isinstance(sources, str) else sources
        except (json.JSONDecodeError, TypeError):
            return "No sources available."
        
        if not sources_data:
            return "No sources available."
        
        formatted_sources = []
        for source in sources_data[:5]:  # Limit to 5 sources to avoid token limits
            title = source.get('title', 'Untitled')
            summary = source.get('summary', 'No summary')[:300]
            highlights = source.get('highlights', [])
            highlights_text = " | ".join(highlights[:3]) if highlights else "No highlights"
            formatted_sources.append(
                f"- **{title}**\n  Summary: {summary}\n  Key excerpts: {highlights_text}"
            )
        return "\n\n".join(formatted_sources)
    
    def simplify(self, text: str, sources: str | None = None) -> str:
        """
        Simplify medical text to patient-friendly language.
        
        Args:
            text: Original medical text to simplify.
            sources: Optional JSON string of sources to cite.
        
        Returns:
            JSON string with simplified_report, key_words, and citations.
        """
        sources_text = self._format_sources(sources)
        
        # Extract medical concepts that MUST be preserved
        concepts_prompt = extract_concepts_for_prompt(text, self.opensearch_client)
        if concepts_prompt:
            logger.debug(f"Extracted concepts for preservation:\n{concepts_prompt}")
        
        prompt = f"""
Simplify the given medical text into easy to understand language for patients, aiming for a 6th grade reading level.
You should also provide key words of the report. Your simplified text should be written in markdown.

{concepts_prompt}

Rules:
- Do not create false information, only simplify what is given.
- Do not include any additional information that is not in the original text.
- With the markdown, you should bold important medical terms and key words. If something can be described in a list format, use bullet points.
- You must mimic the modality of the numbers in the text. For example, if a number is given in digit form (10), keep it in digit form. If a number is given in text form (sixty-seven), keep it in text form.
- PRESERVE ALL medical concepts, diagnoses, measurements, and anatomical references from the original. You may explain them in simpler terms but do NOT omit them.

IMPORTANT - Inline Citation Rules:
- You MUST insert inline citations directly into the simplified_report text using the format [Source Title] at the end of sentences.
- Use the EXACT source title from the sources provided below.
- Place the citation in square brackets immediately after the sentence it supports, before the period. Example: "This disease spreads through mosquitoes [Source Title Here]."
- Try to have at least 3-5 inline citations throughout your report from different sources.
- Only cite facts that are actually supported by the source content.

Here are relevant medical sources to reference:
{sources_text}

JSON Format: 

{{
    "simplified_report": "This sentence is a fact [Exact Source Title]. Another fact here [Another Source Title]. More content without citation if not directly from sources.",
    "key_words": [
        "Key word 1",
        "Key word 2"
    ],
    "citations": {{
        "Exact Source Title": ["This sentence is a fact."],
        "Another Source Title": ["Another fact here."]
    }}
}}

Medical Text: {text}
"""
        return self.client.generate_json(prompt)
    
    def fix(self, original_text: str, simplified_text: str, issues: str, sources: str | None = None) -> str:
        """
        Fix issues in a simplified text based on validation feedback.
        
        Args:
            original_text: The original medical text.
            simplified_text: The problematic simplified version.
            issues: Description of issues to fix.
            sources: Optional JSON string of sources to cite.
        
        Returns:
            JSON string with fixed simplified_report and citations.
        """
        sources_text = self._format_sources(sources)
        
        # Extract medical concepts that MUST be preserved
        concepts_prompt = extract_concepts_for_prompt(original_text, self.opensearch_client)
        
        prompt = f"""
You are a professional medical report writer. Your goal is to simplify the complicated medical text into easy to understand language for patients.
Your simplified text should be written in markdown.

The previous simplification had the following issues:
{issues}

Please fix these issues in your new simplification.

Concepts to preserve:
{concepts_prompt}

Rules:
- Use simple language that anyone can understand, try to maintain an 8th grade reading level.
- Do not leave out any important medical information.
- Do not try to give medical advice.
- Do not create false information, only simplify what is given.
- Do not include any additional information that is not in the original text.
- With the markdown, you should bold important medical terms and key words. If something can be described in a list format, use bullet points.
- You must mimic the modality of the numbers in the text. For example, if a number is given in digit form (10), keep it in digit form. If a number is given in text form (sixty-seven), keep it in text form.
- Your response will be checked for extra or missing entities, numbers, and negations.
- PRESERVE ALL medical concepts, diagnoses, measurements, and anatomical references from the original.

IMPORTANT - Inline Citation Rules:
- You MUST insert inline citations directly into the simplified_report text using the format [Source Title] at the end of sentences.
- Use the EXACT source title from the sources provided below.
- Place the citation in square brackets immediately after the sentence it supports, before the period. Example: "This disease spreads through mosquitoes [Source Title Here]."
- Try to have at least 3-5 inline citations throughout your report from different sources.
- Only cite facts that are actually supported by the source content.

Here are relevant medical sources to reference:
{sources_text}

JSON Format: 

{{
    "simplified_report": "This sentence is a fact [Exact Source Title]. Another fact here [Another Source Title]. More content without citation if not directly from sources.",
    "citations": {{
        "Exact Source Title": ["This sentence is a fact."],
        "Another Source Title": ["Another fact here."]
    }}
}}

Medical Text: {original_text}

Previous Simplification: {simplified_text}
"""
        return self.client.generate_json(prompt)
    
    def repair_with_validation_errors(
        self,
        original_text: str,
        simplified_text: str,
        validation_prompt: str
    ) -> str:
        """
        Repair simplified text using structured validation errors.
        
        Args:
            original_text: The original medical text.
            simplified_text: The problematic simplified version.
            validation_prompt: Structured prompt from PipelineValidationResult.to_repair_prompt().
        
        Returns:
            JSON string with fixed simplified_report.
        """
        # Extract medical concepts that MUST be preserved
        concepts_prompt = extract_concepts_for_prompt(original_text, self.opensearch_client)
        
        prompt = f"""
You are a professional medical report writer. Your previous simplification has validation errors that must be fixed.

{validation_prompt}

{concepts_prompt}

IMPORTANT: You must fix ALL of the above errors. Do not introduce new information. Do not change correct information.

Rules:
- Fix each error according to the suggested fix.
- Use simple language that anyone can understand.
- Do not leave out any important medical information.
- Do not try to give medical advice.
- Do not create false information, only simplify what is given.
- Preserve all numbers, measurements, and units exactly as they appear in the original.
- Preserve laterality (left/right) exactly as in the original.
- PRESERVE ALL medical concepts from the original - you may explain them but do NOT omit them.

JSON Format: 

{{
    "simplified_report": "This will be the fixed simplified report text.",
}}

Original Medical Text: {original_text}
Previous Simplification (with errors): {simplified_text}
"""
        return self.client.generate_json(prompt)
