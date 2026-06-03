from typing import Any

from omegaconf import MISSING
from pydantic import BaseModel


class GNNKGEConfig(BaseModel):
    embedding_dim: int = MISSING
    gnn_class: str
    gnn_kwargs: dict[str, Any]
    score_fn_class: str
    score_fn_kwargs: dict[str, Any] | None = None
