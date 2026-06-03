from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from filtering.core.contracts.knowledge_graph import KnowledgeGraph
    from filtering.core.entities.candidate import InputNodeCandidate, OutputNodeCandidate

T = TypeVar("T")


class NodeFiltering(ABC, Generic[T]):
    def __init__(self, config: T, kg: "KnowledgeGraph | None" = None) -> None:
        self.config = config
        self.kg = kg

    @abstractmethod
    def filter(
        self, candidates: list["InputNodeCandidate"], max_output_candidates: int | None = None
    ) -> list["OutputNodeCandidate"]:
        raise NotImplementedError
