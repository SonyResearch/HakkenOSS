from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings


class NamedConfig(BaseSettings):
    """Base configuration for components with a name and kwargs."""

    name: str
    kwargs: dict[str, Any] = Field(default_factory=dict)
