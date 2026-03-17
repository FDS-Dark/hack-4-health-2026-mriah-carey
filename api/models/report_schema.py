"""Pydantic models for structured medical report data."""

from pydantic import BaseModel
from typing import List


class SupportingDetail(BaseModel):
    """A supporting detail or extra information for a main idea."""
    detail: str


class MainIdea(BaseModel):
    """A main idea with its supporting details."""
    main_idea: str
    details: List[SupportingDetail]


class ParsedReport(BaseModel):
    """The complete parsed medical report structure."""
    main_ideas: List[MainIdea]
