"""SeGAL inference config."""

from pydantic import Field

from hakken_models.core.configs.base_inference import BaseInferenceConfig


class SeGALInferenceConfig(BaseInferenceConfig):
    """Inference config for SeGAL model."""

    encode_batch_size: int = Field(
        default=512,
        description="Batch size for embedder when encoding entities/relations.",
    )
