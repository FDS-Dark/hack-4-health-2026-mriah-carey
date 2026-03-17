"""LLM-based extractor for medical reports using Gemini API."""

import json
import os
from typing import Optional
from google import genai
from dotenv import load_dotenv

from .base import ReportExtractor
from .text_processor import preprocess_report
from models.report_schema import ParsedReport, MainIdea, SupportingDetail

# Load environment variables from .env file
load_dotenv()


class LLMExtractor(ReportExtractor):
    """
    Extract structured information from medical reports using LLM.
    
    Supports Gemini API
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash",
        use_structured_output: bool = True
    ):
        """
        Initialize the LLM extractor.
        
        Args:
            api_key: Gemini API key (defaults to GEMINI_API_KEY env var)
            model: Model name to use
            use_structured_output: Whether to request JSON/structured output
        """
        self.model = model
        self.use_structured_output = use_structured_output
        
        # Get API key from parameter or environment variable
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Please set it as an environment variable "
                "or pass it as a parameter."
            )

        self.client = genai.Client(api_key=api_key)
    
    def _create_extraction_prompt(self, text: str) -> str:
        """
        Create a prompt for extracting main ideas and details from medical report.
        
        Args:
            text: Medical report text
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""You are a medical information extraction system. Analyze the following medical report and extract the main ideas/critical information along with supporting details.

For each main idea, provide:
1. A clear, concise statement of the main idea (e.g., diagnosis, treatment plan, key finding)
2. A list of supporting details that provide additional context, measurements, dates, or related information

Extract information directly from the report. Do not add information that is not present in the text.

Medical Report:
{text}

Extract the main ideas and their supporting details. Return the response as a JSON object with the following structure:
{{
  "main_ideas": [
    {{
      "main_idea": "Primary diagnosis: [diagnosis name]",
      "details": [
        {{"detail": "[supporting detail 1]"}},
        {{"detail": "[supporting detail 2]"}}
      ]
    }},
    {{
      "main_idea": "[another main idea]",
      "details": [
        {{"detail": "[supporting detail]"}}
      ]
    }}
  ]
}}

Return ONLY valid JSON, no additional text or explanation."""
        return prompt
    
    def _parse_json_response(self, response_text: str) -> ParsedReport:
        """
        Parse JSON response from LLM into ParsedReport model.
        
        Args:
            response_text: JSON string from LLM
            
        Returns:
            ParsedReport model
            
        Raises:
            ValueError: If JSON parsing fails or structure is invalid
        """
        # Try to extract JSON if response has markdown code blocks
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            response_text = response_text[start:end].strip()
        elif "```" in response_text:
            start = response_text.find("```") + 3
            end = response_text.find("```", start)
            response_text = response_text[start:end].strip()
        
        # Parse JSON
        try:
            data = json.loads(response_text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {e}\nResponse: {response_text[:500]}")
        
        # Validate and convert to Pydantic model
        try:
            return ParsedReport(**data)
        except Exception as e:
            raise ValueError(f"Failed to create ParsedReport from JSON: {e}\nData: {data}")
    
    def extract(self, text: str) -> ParsedReport:
        """
        Extract main ideas and supporting details from medical report text.
        
        Args:
            text: Medical report text (can be raw or preprocessed)
            
        Returns:
            ParsedReport with extracted main ideas and details
            
        Raises:
            ValueError: If extraction fails
            RuntimeError: If API call fails
        """
        # Preprocess the text
        processed_text = preprocess_report(text)
        
        # Create prompt
        prompt = self._create_extraction_prompt(processed_text)
        
        try:
            # Make API call using Gemini API
            # Use the client to generate content
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={
                    "temperature": 0.1,
                    "response_mime_type": "application/json" if self.use_structured_output else None
                }
            )
            
            # Extract response content
            response_text = response.text
            
            if not response_text:
                raise RuntimeError("Empty response from LLM")
            
            # Parse and return
            return self._parse_json_response(response_text)
            
        except Exception as e:
            if isinstance(e, (ValueError, RuntimeError)):
                raise
            raise RuntimeError(f"LLM API call failed: {e}")
