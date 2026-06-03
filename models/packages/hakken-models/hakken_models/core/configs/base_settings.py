from typing import TypeVar, cast

from omegaconf import DictConfig, OmegaConf
from pydantic_settings import BaseSettings

T = TypeVar("T", bound="HakkenSettings")


class HakkenSettings(BaseSettings):
    @classmethod
    def from_omegaconf(cls: type[T], cfg: DictConfig) -> T:
        """Create TrainKGEConfig from OmegaConf dictConfig."""
        # Convert OmegaConf to dict, resolving interpolations
        cfg_dict = cast(dict, OmegaConf.to_container(cfg, resolve=True))
        return cls(**cfg_dict)
