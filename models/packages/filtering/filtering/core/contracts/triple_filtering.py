from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from filtering.core.contracts.knowledge_graph import KnowledgeGraph
    from filtering.core.entities.candidate import (
        InputTripleCandidate,
        OutputTripleCandidate,
    )

T = TypeVar("T")


class TripleFiltering(ABC, Generic[T]):
    def __init__(self, config: T, kg: "KnowledgeGraph | None" = None):
        self.config = config
        self.graph = kg

    @abstractmethod
    def filter(
        self, candidates: list["InputTripleCandidate"], max_output_candidates: int | None = None
    ) -> list["OutputTripleCandidate"]:
        raise NotImplementedError
