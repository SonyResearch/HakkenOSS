from __future__ import annotations

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from hakken_models.models.segal.config import SeGALConfig

from .train_common import BaseTrainConfig


class TrainSeGALConfig(BaseTrainConfig):
    segal: SeGALConfig = Field(
        ...,
        description="SeGAL model architecture configuration",
    )
    num_negatives: int = Field(
        default=32,
        description="Number of negative samples per positive triple during training.",
    )
    num_negatives_val: int | None = Field(
        default=None,
        description="Number of negative samples per positive during validation. If None, uses num_negatives.",
    )
    learn_embeddings: bool = Field(
        default=False,
        description=(
            "Train entity and relation embedding tables (encoder_dim) end-to-end. "
            "If True, merges into SeGALConfig.learn_embeddings; warm-start from disk when "
            "embeddings exist, otherwise random init. If False, frozen buffers from pre-computed files."
        ),
    )
    embedding_lr_factor: float = Field(
        default=0.1,
        ge=0.0,
        description="Optimizer LR multiplier for embedding tables vs SeGAL body (param groups).",
    )
    embeddings_random_init: bool = Field(
        default=False,
        description=(
            "When learn_embeddings is True, always use random tables (ignore on-disk embeddings). "
            "When False, missing files still use random init; existing files are warm-started if dim matches encoder_dim."
        ),
    )

    model_config = SettingsConfigDict(
        env_prefix="TRAIN_SEGAL_",
        case_sensitive=False,
        extra="ignore",
    )
