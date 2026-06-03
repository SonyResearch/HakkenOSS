from langchain_core.documents import Document
from pydantic import BaseModel, Field

from hakken_agents.utils.file import load_file


class DocumentConfig(BaseModel):
    """Configuration for input file and document metadata."""

    path: str = Field(default="", description="Path to input file")
    encoding: str = Field(default="utf-8", description="Encoding of the input file")
    group_id: str = Field(
        default="agro-kg",
        description="Group identifier for organizing related episodes in the graph",
    )
    source_description: str = Field(
        default="French agricultural/horticultural text",
        description="Human-readable description of the source document",
    )
    reference_year: int = Field(
        default=1880,
        description="Reference year for temporal context of the document",
    )
    source_type: str = Field(
        default="text",
        description="Type of the source document",
    )
    lang: str = Field(
        default="en",
        description="Document language for parsing/OCR (e.g., en, es, ch, ja).",
    )

    def load_document(self) -> Document:
        content = load_file(self.path, self.encoding)
        return Document(
            page_content=content,
            metadata={
                "source_document_path": self.path,
                "group_id": self.group_id,
                "source_description": self.source_description,
                "reference_year": self.reference_year,
                "source_type": self.source_type,
            },
        )
