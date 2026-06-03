from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Sequence

    from simple_query.link_predictor.entities.inputs import LinkPredictorInputTriple

T = TypeVar("T")


class LinkPredictor(ABC, Generic[T]):
    """
    Base class for link predictor implementations.
    It is defined as a generic class, so that the implementation can be coupled with
    its corresponding config class for more comprehensive type annotations.
    """

    def __init__(self, config: T) -> None:
        self.config = config

    def predict(self, triples: "Sequence[LinkPredictorInputTriple]") -> list[float]:
        if not triples:
            return []
        return self._predict(triples)

    @abstractmethod
    def _predict(self, triples: "Sequence[LinkPredictorInputTriple]") -> list[float]:
        raise NotImplementedError
