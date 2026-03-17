"""Gemini client for AI-powered text generation."""

import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from google import genai

load_dotenv()


@dataclass
class GeminiConfig:
    """Configuration for Gemini client."""
    model: str = "gemini-2.5-flash"
    response_mime_type: str | None = None


class GeminiClient:
    """
    Reusable Gemini client wrapper.
    
    Provides a single client instance that can be passed to services
    that need to interact with Gemini models.
    """
    
    def __init__(self, api_key: str | None = None):
        """
        Initialize Gemini client.
        
        Args:
            api_key: Optional API key. If not provided, uses GOOGLE_API_KEY env var.
        """
        if api_key:
            self._client = genai.Client(api_key=api_key)
        else:
            self._client = genai.Client()
    
    @property
    def client(self) -> genai.Client:
        """Get the underlying genai client."""
        return self._client
    
    def generate(
        self,
        prompt: str,
        model: str = "gemini-2.5-flash",
        response_mime_type: str | None = None,
        **kwargs: Any
    ) -> str:
        """
        Generate content using Gemini.
        
        Args:
            prompt: The prompt to send to the model.
            model: Model name to use.
            response_mime_type: MIME type for response (e.g., "application/json").
            **kwargs: Additional config options.
        
        Returns:
            Generated text response.
        """
        config = {}
        if response_mime_type:
            config["response_mime_type"] = response_mime_type
        config.update(kwargs)
        
        response = self._client.models.generate_content(
            model=model,
            contents=prompt,
            config=config if config else None
        )
        return response.text
    
    def generate_json(
        self,
        prompt: str,
        model: str = "gemini-2.5-flash",
        **kwargs: Any
    ) -> str:
        """
        Generate JSON content using Gemini.
        
        Convenience method that sets response_mime_type to application/json.
        
        Args:
            prompt: The prompt to send to the model.
            model: Model name to use.
            **kwargs: Additional config options.
        
        Returns:
            Generated JSON string response.
        """
        return self.generate(
            prompt=prompt,
            model=model,
            response_mime_type="application/json",
            **kwargs
        )


# Singleton instance for convenience
_default_client: GeminiClient | None = None


def get_gemini_client() -> GeminiClient:
    """Get or create the default Gemini client singleton."""
    global _default_client
    if _default_client is None:
        _default_client = GeminiClient()
    return _default_client
