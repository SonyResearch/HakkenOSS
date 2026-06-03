from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from query_common.entities.kg.concept import Concept
    from query_common.entities.kg.identifier import ConceptIdentifier
    from query_common.entities.variable import Variable, VarLabel

ConditionID = int


class Condition(ABC):
    def __init__(self, id_: ConditionID):
        self.id = id_

    @abstractmethod
    def variables(self) -> list["Variable"]:
        """Returns the variable terms in the condition."""
        pass

    @abstractmethod
    def assigned_concepts(self) -> list["Concept"]:
        """Returns the constant concepts in the condition."""
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self!s}>"

    def repr_with_assignment(self, assignment: dict["VarLabel", "ConceptIdentifier"]) -> str:  # noqa: ARG002
        return str(self)


class AtomicCondition(Condition, ABC):
    pass
