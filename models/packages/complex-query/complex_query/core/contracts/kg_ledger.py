from abc import abstractmethod
from typing import TYPE_CHECKING, TypeVar

from complex_query.core.contracts.kg import KnowledgeGraph

if TYPE_CHECKING:
    from collections.abc import Sequence

    from query_common.entities.kg.concept import Concept
    from query_common.entities.kg.identifier import DomainIdentifier
    from query_common.entities.kg.triple import Triple

T = TypeVar("T")


class KnowledgeGraphLedger(KnowledgeGraph[T]):
    @abstractmethod
    def add_concept(self, concept: "Concept") -> None:
        raise NotImplementedError

    @abstractmethod
    def add_triple(self, triple: "Triple") -> None:
        raise NotImplementedError

    @abstractmethod
    def add_concepts_for_domain(
        self, concepts: "Sequence[Concept]", domain_identifier: "DomainIdentifier"
    ) -> None:
        # Add all concepts for the domain, retrievd from an actual KG, to the ledger.
        raise NotImplementedError
