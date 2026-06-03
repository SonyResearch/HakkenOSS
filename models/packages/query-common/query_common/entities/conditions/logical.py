from __future__ import annotations

from typing import TYPE_CHECKING

from query_common.entities.conditions.base import Condition, ConditionID

if TYPE_CHECKING:
    from query_common.entities.kg.concept import Concept
    from query_common.entities.variable import Variable


class _ConnectedCondition(Condition):
    def __init__(self, id_: ConditionID, conditions: list[Condition]):
        super().__init__(id_)
        self.conditions = conditions

    def variables(self) -> list[Variable]:
        all_variables = {}
        for condition in self.conditions:
            for variable in condition.variables():
                all_variables[variable.label] = variable
        return list(all_variables.values())

    def assigned_concepts(self) -> list[Concept]:
        all_concepts = {}
        for condition in self.conditions:
            for variable in condition.assigned_concepts():
                all_concepts[variable.label] = variable
        return list(all_concepts.values())

    def flattened_conditions(self) -> list[Condition]:
        """Flatten nested disjunctive or conjunctive conditions."""
        conditions = []
        for condition in self.conditions:
            if isinstance(condition, self.__class__):
                conditions += condition.flattened_conditions()
            else:
                conditions += [condition]
        return conditions

    def __str__(self):
        raise NotImplementedError


class DisjunctiveCondition(_ConnectedCondition):
    def __str__(self):
        return f"OR({','.join(map(str, self.conditions))})"


class ConjunctiveCondition(_ConnectedCondition):
    def __str__(self):
        return f"AND({','.join(map(str, self.conditions))})"


class NegatedCondition(Condition):
    def __init__(self, id_: ConditionID, condition: Condition):
        super().__init__(id_)
        self.condition = condition

    def variables(self) -> list[Variable]:
        return self.condition.variables()

    def assigned_concepts(self) -> list[Concept]:
        return self.condition.assigned_concepts()

    def __str__(self):
        return f"NOT({self.condition!s})"
