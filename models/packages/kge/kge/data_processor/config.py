from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class KGDataProcessorConfig(BaseModel):
    loader: dict[str, Any] = Field(
        default_factory=lambda: {
            "batch_size": 128,
            "num_workers": 4,
            "persistent_workers": False,
            "pin_memory": False,
        },
        description="Data loader configuration for KGDataProcessor.",
    )
