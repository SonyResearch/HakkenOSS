from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from query_common.entities.kg.triple import Triple

T = TypeVar("T")


class ScoreLedger(ABC, Generic[T]):
    def __init__(self, config: T) -> None:
        self.config = config

    @abstractmethod
    def save_link_score(self, triple: "Triple", score: float) -> None:
        pass

    @abstractmethod
    def retrieve_link_score(self, triple: "Triple") -> float:
        """Raises key error if triple not found"""
        pass
