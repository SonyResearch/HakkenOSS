from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

from pydantic import BaseModel

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    import torch

    from hakken_ml_toolkit.ml_utils.extras.domain import TensorND


class ScalerConfig(BaseModel):
    pass


T = TypeVar("T", bound=ScalerConfig)


class ScalerI(ABC, Generic[T]):
    def __init__(self, config: T):
        self.config = config

    def __str__(self) -> str:
        return f"{self.__class__.__name__}"

    @abstractmethod
    def to(self, device: str | torch.device) -> None:
        pass

    @abstractmethod
    def fit(self, data: TensorND) -> None:
        pass

    @abstractmethod
    def fit_from_iterator(self, iterator: Iterator, num_batches: int | None = None) -> None:
        pass

    @abstractmethod
    def transform(self, data: TensorND) -> TensorND:
        pass

    def fit_transform(self, data: TensorND) -> TensorND:
        self.fit(data)
        return self.transform(data)

    @abstractmethod
    def inverse_transform(self, data_norm: TensorND) -> TensorND:
        pass

    @abstractmethod
    def save(self, json_path: str | Path):
        pass

    @classmethod
    @abstractmethod
    def load(cls, json_path: str) -> ScalerI:
        pass
