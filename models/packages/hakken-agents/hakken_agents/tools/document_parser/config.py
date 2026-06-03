from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from hakken_agents.tools.document_parser.enums import ParseMethodType, ParserType


class ParseDocumentConfig(BaseModel):
    """Configuration for document parsing operations."""

    file_path: str = Field(
        default="",
        description="Path to the document file to parse.",
    )
    output_dir: str = Field(
        default="./output",
        description="Directory for parser output (markdown, JSON, extracted images).",
    )
    parse_method: ParseMethodType = Field(
        default=ParseMethodType.AUTO,
        description="Parsing method: auto (detect), ocr (optical character recognition), txt (text extraction).",
    )
    display_stats: bool = Field(
        default=True,
        description="Whether to log content block statistics after parsing.",
    )
    parser: ParserType = Field(
        default=ParserType.MINERU,
        description="Parser backend: MinerU (PDF/images/Office) or Docling (PDF/Office/HTML).",
    )
    max_chunk_size: int = Field(
        default=1000,
        ge=1,
        description="Maximum character size for text chunks when splitting parsed content.",
    )
    # Parser-specific options (passed to MinerU/Docling)
    lang: str | None = Field(
        default=None,
        description="Document language for OCR optimization (e.g., ch, en, ja).",
    )
    device: str | None = Field(
        default=None,
        description="Inference device (e.g., cpu, cuda, cuda:0, npu, mps).",
    )
    start_page: int | None = Field(
        default=None,
        description="PDF start page (0-based). None = first page.",
    )
    end_page: int | None = Field(
        default=None,
        description="PDF end page (0-based). None = last page.",
    )
    formula: bool = Field(
        default=True,
        description="Enable formula/equation parsing.",
    )
    table: bool = Field(
        default=True,
        description="Enable table parsing.",
    )
    backend: (
        Literal[
            "pipeline",
            "hybrid-auto-engine",
            "hybrid-http-client",
            "vlm-auto-engine",
            "vlm-http-client",
        ]
        | None
    ) = Field(
        default=None,
        description="MinerU parsing backend. None uses parser default.",
    )
    source: Literal["huggingface", "modelscope", "local"] | None = Field(
        default=None,
        description="Model source for MinerU. None uses parser default.",
    )
    vlm_url: str | None = Field(
        default=None,
        description="VLM server URL when backend is vlm-http-client (e.g., http://127.0.0.1:30000).",
    )

    def get_parser_kwargs(self, **overrides: Any) -> dict[str, Any]:
        """Build parser kwargs from config, with overrides taking precedence."""
        kwargs: dict[str, Any] = {}
        if self.lang is not None:
            kwargs["lang"] = self.lang
        if self.device is not None:
            kwargs["device"] = self.device
        if self.start_page is not None:
            kwargs["start_page"] = self.start_page
        if self.end_page is not None:
            kwargs["end_page"] = self.end_page
        kwargs["formula"] = self.formula
        kwargs["table"] = self.table
        if self.backend is not None:
            kwargs["backend"] = self.backend
        if self.source is not None:
            kwargs["source"] = self.source
        if self.vlm_url is not None:
            kwargs["vlm_url"] = self.vlm_url
        kwargs.update(overrides)
        return kwargs
