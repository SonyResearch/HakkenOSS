from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import torch
from hakken_ml_toolkit.ml_base_structures import KnowledgeGraph
from pydantic import BaseModel


class TripleFiltererConfig(BaseModel):
    pass


T = TypeVar("T", bound=TripleFiltererConfig)


class TripleFilterI(ABC, Generic[T]):
    def __init__(self, kg: KnowledgeGraph, config: T | None = None):
        """
        Initializes the TripleFilterer with the provided configuration.

        Args:
            config (TripleFiltererConfig): The configuration containing filter criteria.
            kg (Optional[KnowledgeGraph]): The knowledge graph to be used for filtering.
        """
        self.config = config
        self.kg = kg

    def set_up(self, kg: KnowledgeGraph) -> None:
        """
        Sets up the TripleFilterer with the knowledge graph.

        Args:
            kg (KnowledgeGraph): The knowledge graph to be used for filtering.
        """
        self.kg = kg

    @abstractmethod
    def compute_scores(
        self, sro_batch: torch.Tensor, scores: torch.Tensor, inplace: bool = True
    ) -> torch.Tensor:
        pass
