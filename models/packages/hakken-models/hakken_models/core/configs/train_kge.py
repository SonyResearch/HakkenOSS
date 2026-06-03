from pydantic import Field
from pydantic_settings import SettingsConfigDict

from .model import KGEConfig
from .negative_strategy import NegativeStrategyConfig
from .train_common import BaseTrainConfig, NegSamplerConfig


class TrainKGEConfig(BaseTrainConfig):
    negative_strategy: NegativeStrategyConfig = Field(
        default_factory=NegativeStrategyConfig,
        description="Configuration for negative sampling strategy",
    )
    kge: KGEConfig = Field(default_factory=KGEConfig)
    negative_sampler: NegSamplerConfig = Field(default_factory=NegSamplerConfig)

    model_config = SettingsConfigDict(
        env_prefix="TRAIN_KGE_",
        case_sensitive=False,
        extra="ignore",
    )
