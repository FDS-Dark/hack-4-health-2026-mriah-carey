"""API request and response models."""

from pydantic import BaseModel
from typing import Dict, Any
from models.report_schema import ParsedReport
from models.simplified_schema import SimplifiedReport


class ProcessingResponse(BaseModel):
    """Complete API response with all processing results."""
    structured_json: ParsedReport
    simplified_json: SimplifiedReport
    simplified_text: str
    summary_text: str
    metadata: Dict[str, Any]  # Processing stats, timestamps, etc.
