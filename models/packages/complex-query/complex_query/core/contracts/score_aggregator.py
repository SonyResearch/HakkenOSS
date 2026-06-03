import functools
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

from complex_query.core.values.errors import ScoreAggregatorError

if TYPE_CHECKING:
    from collections.abc import Sequence

T = TypeVar("T")


class ScoreAggregator(ABC, Generic[T]):
    def __init__(self, config: T) -> None:
        self.config = config

    @abstractmethod
    def binary_t_norm(self, a: float, b: float) -> float:
        raise NotImplementedError

    def binary_t_conorm(self, a: float, b: float) -> float:
        return 1 - self.binary_t_norm(1 - a, 1 - b)

    def t_norm(self, inputs: "Sequence[float]") -> float:
        if not inputs:
            raise ScoreAggregatorError("Inputs to t_norm is empty.")
        return functools.reduce(self.binary_t_norm, inputs)

    def t_conorm(self, inputs: "Sequence[float]") -> float:
        if not inputs:
            raise ScoreAggregatorError("Inputs to t_norm is empty.")
        return functools.reduce(self.binary_t_conorm, inputs)
