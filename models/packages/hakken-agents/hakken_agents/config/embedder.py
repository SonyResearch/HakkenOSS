from __future__ import annotations

from pydantic import BaseModel, Field, SecretStr, field_validator


class EmbedderConfig(BaseModel):
    """Configuration for the embedding model."""

    api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(""),
        description="API key for embedding service",
    )

    @field_validator("api_key", mode="before")
    @classmethod
    def _coerce_api_key(cls, v: object) -> SecretStr:
        if v is None or v == "":
            return SecretStr("")
        if isinstance(v, str):
            return SecretStr(v)
        return v

    base_url: str | None = Field(default=None, description="Base URL for embedding API")
    embedding_model: str = Field(
        default="text-embedding-3-small", description="Embedding model name"
    )
    embedding_dim: int = Field(default=1536, description="Embedding dimension size")
