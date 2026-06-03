from __future__ import annotations

from typing import TYPE_CHECKING

from query_common.entities.conditions.base import AtomicCondition, ConditionID
from query_common.entities.kg.concept import Concept
from query_common.entities.variable import Variable

if TYPE_CHECKING:
    from query_common.entities.kg.relation import Relation


class NoveltyCondition(AtomicCondition):
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
        return f"Novel({s_label}, {self.relation.label}, {o_label})"
