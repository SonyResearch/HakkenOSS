from enum import StrEnum

from pydantic import BaseModel, Field


class TextSplitterKind(StrEnum):
    """Text splitter implementation kind."""

    RC = "recursive_char"  # RecursiveCharacterTextSplitter


class TextSplitterConfig(BaseModel):
    """Configuration for text chunking."""

    kind: TextSplitterKind = Field(
        default=TextSplitterKind.RC,
        description="Text splitter kind (e.g. 'recursive_char' for RecursiveCharacterTextSplitter)",
    )
    chunk_size: int = Field(default=1_000, description="Maximum size of each text chunk")
    chunk_overlap: int = Field(default=150, description="Overlap between consecutive chunks")
    separators: list[str] = Field(
        default=["\n\n", "\n", ". ", " ", ""],
        description="Separator hierarchy for splitting",
    )
