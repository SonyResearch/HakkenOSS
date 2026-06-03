from pydantic import BaseModel, Field

from hakken_agents.enki.schemas.fact import Fact


class ExtractedFacts(BaseModel):
    """Container for all extracted facts from text."""

    facts: list[Fact] = Field(description="List of extracted facts")
