from typing import Any

from pydantic import BaseModel, Field


class KGEConfig(BaseModel):
    embedding_dim: int = 128
    score_fn_name: str = "ComplExScore"
    score_fn_kwargs: dict[str, Any] = Field(default_factory=dict)


class GNNConfig(BaseModel):
    name: str = "GraphSAGE"
    kwargs: dict[str, Any] = Field(
        default_factory=lambda: {
            "hidden_channels": 64,
            "num_layers": 2,
            "dropout": 0.0,
            "act": "relu",
        }
    )


class TransformerConfig(BaseModel):
    name: str = "Transformer"
    kwargs: dict[str, Any] = Field(
        default_factory=lambda: {
            "num_heads": 8,
            "num_layers": 2,
            "dropout": 0.1,
            "use_pos_encoding": True,
            "aggregation": "cls_token",
        }
    )


class THiGERConfig(BaseModel):
    entity_embedding_dim: int = 64
    relation_embedding_dim: int = 64
    domain_embedding_dim: int | None = None
    has_logits: bool = True
    gnn: GNNConfig = Field(default_factory=lambda: GNNConfig())
    transformer: TransformerConfig = Field(default_factory=lambda: TransformerConfig())
