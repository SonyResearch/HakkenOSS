from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

from pydantic import TypeAdapter
from query_common.entities.kg.identifier import (
    ConceptIdentifier,
    DomainIdentifier,
    RelationIdentifier,
)

if TYPE_CHECKING:
    from query_common.entities.kg.concept import Concept
    from query_common.entities.kg.triple import Triple

_concept_identifier_adapter = TypeAdapter(ConceptIdentifier)
_relation_identifier_adapter = TypeAdapter(RelationIdentifier)

T = TypeVar("T")


class KnowledgeGraph(ABC, Generic[T]):
    def __init__(self, config: T) -> None:
        self.config = config

    @abstractmethod
    def add_concept(self, node: "Concept"):
        """
        Add a node to the graph.
        """
        pass

    @abstractmethod
    def add_triple(self, triple: "Triple"):
        """
        Add a triple to the graph.
        """
        pass

    def get_concept(self, node_identifier: "ConceptIdentifier | str") -> "Concept":
        node_identifier = _concept_identifier_adapter.validate_python(node_identifier)
        return self._get_concept(node_identifier)

    @abstractmethod
    def _get_concept(self, node_identifier: "ConceptIdentifier") -> "Concept":
        """
        Retrieve a node by its ID.
        """
        pass

    @abstractmethod
    def get_concepts_from_domain(self, domain_identifier: "DomainIdentifier") -> list["Concept"]:
        """
        Retrieve all concepts belonging to the given domain.
        """
        pass

    def get_triples(
        self,
        subject_identifier: "ConceptIdentifier | str | None" = None,
        object_identifier: "ConceptIdentifier | str | None" = None,
        relation_identifier: "RelationIdentifier | str | None" = None,
    ) -> list["Triple"]:
        if subject_identifier is not None:
            subject_identifier = _concept_identifier_adapter.validate_python(subject_identifier)
        if object_identifier is not None:
            object_identifier = _concept_identifier_adapter.validate_python(object_identifier)
        if relation_identifier is not None:
            relation_identifier = _relation_identifier_adapter.validate_python(relation_identifier)
        return self._get_triples(
            subject_identifier=subject_identifier,
            object_identifier=object_identifier,
            relation_identifier=relation_identifier,
        )

    @abstractmethod
    def _get_triples(
        self,
        subject_identifier: "ConceptIdentifier | None" = None,
        object_identifier: "ConceptIdentifier | None" = None,
        relation_identifier: "RelationIdentifier | None" = None,
    ) -> list["Triple"]:
        """
        Retrieve triples in the graph that match the query.
        """
        pass
