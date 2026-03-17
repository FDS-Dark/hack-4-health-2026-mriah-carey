"""Report compilation services for converting simplified JSON to human-readable text."""

import json
import os
from abc import ABC, abstractmethod
from typing import Optional
from google import genai
from dotenv import load_dotenv

from models.simplified_schema import SimplifiedReport

load_dotenv()


class ReportCompiler(ABC):
    """Abstract base class for compiling simplified reports to human-readable text."""
    
    @abstractmethod
    def compile(self, simplified_report: SimplifiedReport, format: str = "paragraphs") -> str:
        """
        Compile simplified report to human-readable text.
        
        Args:
            simplified_report: The simplified report to compile
            format: Output format - "paragraphs" or "sections"
            
        Returns:
            Human-readable text
        """
        pass


class LLMCompiler(ReportCompiler):
    """
    LLM-based compiler that creates paragraphs/sections from simplified JSON.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash"
    ):
        """
        Initialize the LLM compiler.
        
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
    
    def _create_compilation_prompt(
        self,
        simplified_report: SimplifiedReport,
        format: str
    ) -> str:
        """
        Create a prompt for compiling simplified report to human-readable format.
        
        Args:
            simplified_report: The simplified report
            format: Output format - "paragraphs" or "sections"
            
        Returns:
            Formatted prompt string
        """
        # Extract only simplified versions for compilation
        simplified_content = []
        for main_idea in simplified_report.simplified_main_ideas:
            content_item = {
                "main_idea": main_idea.simplified_main_idea,
                "details": [detail.simplified_detail for detail in main_idea.simplified_details]
            }
            simplified_content.append(content_item)
        
        # Convert to JSON string for the prompt
        simplified_json = json.dumps(simplified_content, indent=2)
        
        format_instructions = ""
        if format == "paragraphs":
            format_instructions = """
- Create flowing, continuous, well-formatted paragraphs
- Connect all main points and details naturally with smooth transitions
- Write as a cohesive narrative that reads smoothly from start to finish
- Tie all information together into coherent paragraphs
- Do not use headings or section breaks
- Ensure proper paragraph formatting with appropriate line breaks
- Make it read like a natural, well-written medical report summary
"""
        elif format == "sections":
            format_instructions = """
- Create structured sections with clear headings
- Each main idea should be its own section with a descriptive heading
- Use formatting like "## [Section Title]" for headings
- Write clear, readable paragraphs within each section
- Tie details to their main ideas naturally
"""
        
        prompt = f"""You are a medical report writer. Your task is to compile simplified medical information into a clear, well-formatted, human-readable format that patients can easily understand.

The simplified information is provided in JSON format below. Each entry has a main idea and supporting details, all already simplified to plain language. Convert this into a well-written, readable medical report.

CRITICAL REQUIREMENTS:
- Use ONLY the simplified_main_idea and simplified_details from the JSON (these are already in plain language)
- Create well-formatted, coherent paragraphs that tie all main points and details together
- Ensure proper paragraph structure with appropriate spacing
- Connect ideas naturally with smooth transitions
- Write in a professional but accessible tone
- Maintain all information from the simplified entries

Format Requirements:
{format_instructions}
- Ensure the text flows naturally and is easy to read
- All main points and details should be included and well-integrated

Simplified Medical Information (JSON):
{simplified_json}

Generate the compiled report with well-formatted paragraphs now:"""
        
        return prompt
    
    def compile(self, simplified_report: SimplifiedReport, format: str = "paragraphs") -> str:
        """
        Compile simplified report to human-readable text.
        
        Args:
            simplified_report: The simplified report to compile
            format: Output format - "paragraphs" or "sections" (default: "paragraphs")
            
        Returns:
            Human-readable text
        """
        if format not in ["paragraphs", "sections"]:
            raise ValueError(f"Format must be 'paragraphs' or 'sections', got '{format}'")
        
        # Create compilation prompt
        prompt = self._create_compilation_prompt(simplified_report, format)
        
        try:
            # Make API call
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={
                    "temperature": 0.3  # Slightly higher for more natural writing
                }
            )
            
            compiled_text = response.text
            
            if not compiled_text:
                raise RuntimeError("Empty response from LLM compiler")
            
            return compiled_text.strip()
            
        except Exception as e:
            raise RuntimeError(f"LLM compilation failed: {e}")
