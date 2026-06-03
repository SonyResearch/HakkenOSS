from __future__ import annotations

from typing import TYPE_CHECKING

from query_common.entities.conditions.base import AtomicCondition, ConditionID
from query_common.entities.kg.concept import Concept
from query_common.entities.variable import Variable, VarLabel

if TYPE_CHECKING:
    from query_common.entities.kg.identifier import ConceptIdentifier
    from query_common.entities.kg.relation import Relation


class LinkCondition(AtomicCondition):
    def __init__(
        self,
        id_: ConditionID,
        subject: Concept | Variable,
        relation: Relation,
        object_: Concept | Variable,
    ):
        super().__init__(id_)
        self.subject = subject
        self.relation = relation
        self.object = object_

    def variables(self) -> list[Variable]:
        return [x for x in [self.subject, self.object] if isinstance(x, Variable)]

    def assigned_concepts(self) -> list[Concept]:
        return [x for x in [self.subject, self.object] if isinstance(x, Concept)]

    def __str__(self):
        s_label = (
            self.subject.identifier if isinstance(self.subject, Concept) else self.subject.label
        )
        o_label = self.object.identifier if isinstance(self.object, Concept) else self.object.label
        return f"Pr({s_label}, {self.relation.label}, {o_label})"

    def repr_with_assignment(self, assignment: dict[VarLabel, ConceptIdentifier]) -> str:
        if isinstance(self.subject, Concept):
            s_label = str(self.subject.identifier)
        else:
            s_label = (
                f"{self.subject.label}={assignment[self.subject.label]}"
                if self.subject.label in assignment
                else self.subject.label
            )
        if isinstance(self.object, Concept):
            o_label = str(self.object.identifier)
        else:
            o_label = (
                f"{self.object.label}={assignment[self.object.label]}"
                if self.object.label in assignment
                else self.object.label
            )
        return f"Pr({s_label}, {self.relation.label}, {o_label})"
