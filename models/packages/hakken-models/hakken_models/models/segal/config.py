"""SeGAL configuration dataclasses."""

from pydantic import BaseModel, Field

from hakken_models.core.configs.model import GNNConfig
from hakken_models.core.embedder import EmbedderConfig


class TemporalEncoderConfig(BaseModel):
    embedding_dim: int = 64
    learnable_frequencies: bool = True
    num_sinusoidal: int | None = None


class ScoringConfig(BaseModel):
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1


class SeGALConfig(BaseModel):
    encoder_dim: int
    embedder: EmbedderConfig
    gnn: GNNConfig = Field(default_factory=GNNConfig)
    temporal: TemporalEncoderConfig = Field(default_factory=TemporalEncoderConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    edge_feature_mode: str = Field(
        default="cat",
        description="How to combine relation and temporal edge features: 'cat' or 'add'.",
    )
    use_inverse_relations: bool = Field(
        default=True,
        description="If True, reverse edges (direction column 1 in edge_attr) use r_emb + inv_emb for GNN edge features.",
    )
    learn_embeddings: bool = Field(
        default=False,
        description=(
            "If True, base entity/relation vectors live in encoder_dim space and are trained "
            "in Lightning (see TrainSeGALConfig.learn_embeddings). input_proj matches that path."
        ),
    )
