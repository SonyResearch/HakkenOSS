"""API configuration - use with Hydra to load from YAML (supports ${env:VAR} interpolation)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from hakken_agents.tools.element_resolver.config import ElementResolverConfig


class ElementResolverAPIConfig(BaseModel):
    """Configuration for the Element Resolver API.

    Load via Hydra from YAML; use ${env:VAR} for secrets and env interpolation.
    """

    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8000, ge=1, le=65535, description="API port")
    resolver: ElementResolverConfig = Field(
        description="Element resolver configuration (LLM, DB, embedder, table)"
    )
