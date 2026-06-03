from collections.abc import Sequence
from enum import Enum

from pydantic import BaseModel

from contextualization.core.entities.publication import Publication


class Reference(BaseModel):
    publication_info: Publication
    score: float
    text: str | None = None
    summary: str | None = None


class RetrievedContext(BaseModel):
    references: Sequence[Reference]
    summary: str | None = None


class RetrievalReturnType(Enum):
    PUBLICATION = "publication"
    TEXT = "text"
    SUMMARY = "summary"
