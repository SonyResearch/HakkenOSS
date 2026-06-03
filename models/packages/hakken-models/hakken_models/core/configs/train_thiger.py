from pydantic import Field
from pydantic_settings import SettingsConfigDict

from .model import THiGERConfig
from .train_common import BaseTrainConfig


class TrainTHiGERConfig(BaseTrainConfig):
    thiger: THiGERConfig = Field(default_factory=lambda: THiGERConfig())

    model_config = SettingsConfigDict(
        env_prefix="TRAIN_THIGER_",
        case_sensitive=False,
        extra="ignore",
    )
