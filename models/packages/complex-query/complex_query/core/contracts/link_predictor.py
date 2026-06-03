from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray
    from query_common.entities.kg.triple import Triple

T = TypeVar("T")


class LinkPredictor(ABC, Generic[T]):
    def __init__(self, config: T) -> None:
        self.config = config

    @abstractmethod
    def predict(self, triple: "Triple") -> float:
        pass

    def predict_batch(self, triples: "Sequence[Triple]") -> "NDArray[np.float64]":
        return np.array([self.predict(t) for t in triples])
