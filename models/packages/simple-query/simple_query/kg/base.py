from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from query_common.entities.kg.identifier import ConceptIdentifier, DomainIdentifier

    from simple_query.kg.entities.constraint import ConstraintFilteringOutput, TripleConstraint
    from simple_query.query.entities.inputs import ConditionNode

T = TypeVar("T")


class KnowledgeGraph(ABC, Generic[T]):
    """
    Base class for KG implementations.
    It is defined as a generic class, so that the implementation can be coupled with
    its corresponding config class for more comprehensive type annotations.
    """

    def __init__(self, config: T) -> None:
        self.config = config

    @abstractmethod
    def get_concept_identifiers(
        self, domain_identifier: "DomainIdentifier | None", condition: "ConditionNode | None"
    ) -> list["ConceptIdentifier"]:
        raise NotImplementedError

    @abstractmethod
    def filter_constraint(
        self, triple_constraint: "TripleConstraint"
    ) -> "ConstraintFilteringOutput":
        raise NotImplementedError
