from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from pydantic import BaseModel
from torch import Tensor, nn


class InitStrategyConfig(BaseModel):
    """Base configuration for initialization strategies."""

    pass


ConfigType = TypeVar("ConfigType", bound=InitStrategyConfig)
StrategyType = TypeVar("StrategyType", bound="BaseInitStrategy")


class BaseInitStrategy(ABC, Generic[ConfigType]):
    def __call__(self, model: nn.Module) -> None:
        for _name, module in model.named_modules():
            if isinstance(module, nn.Conv1d | nn.Conv2d | nn.Conv3d):
                self.init_conv(module)
            elif isinstance(module, nn.Linear):
                self.init_linear(module)
            elif isinstance(module, nn.Embedding):
                self.init_embedding(module)
            # Skip normalization layers - they have their own initialization
            else:
                continue

    @abstractmethod
    def init_conv(self, module: nn.Conv1d | nn.Conv2d | nn.Conv3d) -> None:
        """Initialize convolutional layer weights and biases."""
        pass

    @abstractmethod
    def init_linear(self, module: nn.Linear) -> None:
        """Initialize linear layer weights and biases."""
        pass

    @abstractmethod
    def init_embedding(self, module: nn.Embedding) -> None:
        """Initialize embedding layer weights."""
        pass

    def _init_bias(self, bias: Tensor | None) -> None:
        """Default bias initialization - can be overridden by subclasses."""
        if bias is not None:
            nn.init.zeros_(bias)

    @classmethod
    def from_config(cls: type[StrategyType], config: ConfigType) -> StrategyType:
        return cls(**config.model_dump())
