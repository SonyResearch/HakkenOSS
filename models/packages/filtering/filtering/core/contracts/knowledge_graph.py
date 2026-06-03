from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from filtering.core.entities.kg import EdgeDirection, NodeId, YearRange

T = TypeVar("T")


class KnowledgeGraph(ABC, Generic[T]):
    def __init__(self, config: T):
        self.config = config

    @abstractmethod
    def get_degrees(
        self,
        node_ids: list["NodeId"],
        direction: "EdgeDirection",
        year_range: "YearRange | None" = None,
    ) -> list[int]:
        raise NotImplementedError
