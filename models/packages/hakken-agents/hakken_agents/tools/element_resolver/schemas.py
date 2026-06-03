from typing import Any

from pydantic import BaseModel, Field


class SimilaritySearchParam(BaseModel):
    k: int = Field(default=5, ge=1, description="Number of similar documents to return")
    threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score (0-1) to include a result. None means no filtering.",
    )
    filter: dict[str, Any] | None = Field(
        default=None, description="Optional metadata filter for the search"
    )


# ---------------------------------------------------------------------------
# API request / response schemas
# ---------------------------------------------------------------------------


class ElementPayload(BaseModel):
    """Single element for ingestion: free-form content plus arbitrary metadata."""

    content: str = Field(description="Text content for embedding and search")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata stored alongside the element",
    )


class ElementIngestRequest(BaseModel):
    """Request body for element ingestion."""

    elements: list[ElementPayload] = Field(description="Elements to ingest")
    no_description: bool = Field(
        default=False,
        description="Skip LLM description generation; embed raw content only",
    )
    max_concurrency: int = Field(
        default=5, ge=1, description="Max concurrent LLM calls for description generation"
    )


class ElementIngestResponse(BaseModel):
    """Response for element ingestion."""

    ingested: list[str] = Field(description="UUIDs of newly ingested elements")
    skipped: list[str] = Field(description="UUIDs of elements already present")


class ElementSearchRequest(BaseModel):
    """Request body for element similarity search."""

    query: str = Field(description="Search query text")
    k: int = Field(default=5, ge=1, le=100, description="Number of results to return")
    threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score (0-1) to include a result",
    )
    filter: dict[str, Any] | None = Field(
        default=None,
        description='Optional metadata filter (e.g. {"context": {"$ilike": "%GENE%"}})',
    )


class ElementSearchResult(BaseModel):
    """Single element search result."""

    element_id: str = Field(description="Element UUID")
    score: float = Field(description="Similarity score (1.0 - distance; higher = more similar)")
    content: str = Field(description="Document content")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Element metadata")


class ElementSearchResponse(BaseModel):
    """Response for element similarity search."""

    results: list[ElementSearchResult] = Field(description="Search results")
