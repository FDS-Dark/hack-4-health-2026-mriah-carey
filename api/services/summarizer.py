"""Report summarization services for creating shortened summaries from simplified JSON."""

import json
import os
from abc import ABC, abstractmethod
from typing import Optional
from google import genai
from dotenv import load_dotenv

from models.simplified_schema import SimplifiedReport

load_dotenv()

class ReportSummarizer(ABC):
    """Abstract base class for summarizing simplified reports."""
    
    @abstractmethod
    def summarize(self, simplified_report: SimplifiedReport) -> str:
        """
        Generate shortened summary from simplified report.
        
        Args:
            simplified_report: The simplified report to summarize
            
        Returns:
            Concise summary text
        """
        pass


class LLMSummarizer(ReportSummarizer):
    """
    LLM-based summarizer that creates concise summaries from simplified JSON.
    
    Takes simplified JSON as input to limit AI drift and ensure accuracy.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash"
    ):
        """
        Initialize the LLM summarizer.
        
        Args:
            api_key: Gemini API key (defaults to GEMINI_API_KEY env var)
            model: Model name to use
        """
        self.model = model
        
        # Get API key
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Please set it as an environment variable "
                "or pass it as a parameter."
            )
        
        self.client = genai.Client(api_key=api_key)
    
    def _create_summarization_prompt(
        self,
        simplified_report: SimplifiedReport
    ) -> str:
        """
        Create a prompt for summarizing the simplified report.
        
        Args:
            simplified_report: The simplified report to summarize
            
        Returns:
            Formatted prompt string
        """
        # Extract only simplified versions for summarization
        simplified_content = []
        for main_idea in simplified_report.simplified_main_ideas:
            content_item = {
                "main_idea": main_idea.simplified_main_idea,
                "details": [detail.simplified_detail for detail in main_idea.simplified_details]
            }
            simplified_content.append(content_item)
        
        # Convert to JSON string for the prompt
        simplified_json = json.dumps(simplified_content, indent=2)
        
        prompt = f"""You are a medical report writer. Your task is to create a concise, intentionally short summary from the simplified medical information provided in JSON format below.

CRITICAL REQUIREMENTS:
- Use ONLY the simplified_main_idea and simplified_details from the JSON (these are already in plain language)
- Create a brief summary that is approximately 30-50% of the original length
- Focus ONLY on the most important findings, diagnoses, and critical information
- Intentionally keep it short - remove less critical details and redundant information
- Maintain all essential medical details (diagnoses, key findings, important measurements)
- Use simple, clear language that is easy to understand
- Write as flowing paragraphs (not bullet points) for readability
- Do not add information that is not in the provided simplified entries
- Do not omit critical medical findings or diagnoses

The summary should be significantly shorter than the full simplified text while preserving all key medical information. Focus on what's most important for the patient to know.

Simplified Medical Information (JSON):
{simplified_json}

Generate the concise, intentionally short summary now:"""
        
        return prompt
    
    def summarize(self, simplified_report: SimplifiedReport) -> str:
        """
        Generate shortened summary from simplified report.
        
        Args:
            simplified_report: The simplified report to summarize
            
        Returns:
            Concise summary text
        """
        # Create summarization prompt
        prompt = self._create_summarization_prompt(simplified_report)
        
        try:
            # Make API call
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={
                    "temperature": 0.2  # Lower temperature for more focused summarization
                }
            )
            
            summary_text = response.text
            
            if not summary_text:
                raise RuntimeError("Empty response from LLM summarizer")
            
            return summary_text.strip()
            
        except Exception as e:
            raise RuntimeError(f"LLM summarization failed: {e}")
