"""Pydantic models for simplified medical report data."""

from pydantic import BaseModel
from typing import List


class SimplifiedDetail(BaseModel):
    """A detail with both original and simplified versions."""
    original_detail: str
    simplified_detail: str


class SimplifiedMainIdea(BaseModel):
    """A main idea with both original and simplified versions."""
    original_main_idea: str
    simplified_main_idea: str
    original_details: List[str]  # List of original detail strings
    simplified_details: List[SimplifiedDetail]  # List with original/simplified pairs


class SimplifiedReport(BaseModel):
    """The simplified report structure with original to simplified mapping."""
    simplified_main_ideas: List[SimplifiedMainIdea]
