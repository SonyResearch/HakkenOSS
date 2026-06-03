from pydantic import BaseModel, Field

from hakken_agents.enki.schemas.entity import Entity


class ExtractedEntities(BaseModel):
    """Container for all extracted entities from text."""

    entities: list[Entity] = Field(description="List of extracted entities")
