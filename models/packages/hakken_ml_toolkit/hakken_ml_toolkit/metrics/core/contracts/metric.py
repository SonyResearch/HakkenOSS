from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar, cast

from pydantic import BaseModel

from hakken_ml_toolkit.metrics.core.exceptions import UnknownReductionError

if TYPE_CHECKING:
    from hakken_ml_toolkit.metrics.core.entities import FloatTensor1D


class MetricConfig(BaseModel):
    reduce: str = "sum"


T = TypeVar("T", bound=MetricConfig)


class MetricI(ABC, Generic[T]):
    DEFAULT_CONFIG: ClassVar[T]  # type: ignore
    name: str

    def __init__(self, config: T | dict[str, Any] | None = None):
        self.config = self.DEFAULT_CONFIG
        if isinstance(config, dict):
            self.config = self._get_config_class()(
                **{k: v for k, v in config.items() if k in self._get_config_class().model_fields}
            )
        elif config:
            self.config = config

    @classmethod
    def _get_config_class(cls) -> type[T]:
        return cast("type[T]", cls.DEFAULT_CONFIG.__class__)

    def __call__(self, **kwargs: Any) -> FloatTensor1D:
        self.update(**kwargs)
        return self.compute()

    def reduce(self, values: FloatTensor1D) -> FloatTensor1D:
        if self.config.reduce == "sum":
            data = values.sum().unsqueeze(0)
        elif self.config.reduce == "mean":
            data = values.mean().unsqueeze(0)
        elif self.config.reduce == "none":
            data = values
        else:
            raise UnknownReductionError()

        return data

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def update(self, *args: Any, **kwargs: Any) -> None:
        """
        Update metric state with new inputs.
        """
        pass

    @abstractmethod
    def compute(self) -> FloatTensor1D:
        pass
