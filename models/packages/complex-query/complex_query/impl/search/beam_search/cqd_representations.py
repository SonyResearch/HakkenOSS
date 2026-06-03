from __future__ import annotations

from typing import TYPE_CHECKING

from query_common.entities.query import Candidate

from complex_query.impl.search.beam_search.generic.problem import PartialSolution, Step

if TYPE_CHECKING:
    from query_common.entities.conditions.base import Condition
    from query_common.entities.kg.identifier import ConceptIdentifier
    from query_common.entities.variable import VarLabel


class QueryConditionStep(Step):
    """A step is evaluating a new condition with a given variable assignment"""

    def __init__(self, condition: Condition, assignment: dict[VarLabel, ConceptIdentifier]):
        self.condition = condition
        self.assignment = assignment

    def __str__(self):
        return f"Use assignment {self.assignment} to solve condition {self.condition}"

    def short_repr(self) -> str:
        return self.condition.repr_with_assignment(self.assignment)


class QueryPartialSolution(PartialSolution):
    def __init__(self, candidate: Candidate):
        self.candidate = candidate

    def __str__(self):
        return str(self.candidate)

    def short_repr(self) -> str:
        return str(self.candidate.var_assignments)

    @classmethod
    def from_empty(cls):
        return QueryPartialSolution(candidate=Candidate(var_assignments={}, condition_scores={}))

    def get_assignment_for_variable(self, var_label: VarLabel) -> ConceptIdentifier:
        try:
            return self.candidate.var_assignments[var_label]
        except KeyError as e:
            raise KeyError(f"Variable label '{var_label}' not found in partial solution") from e

    def is_variable_already_assigned(self, var_label: VarLabel) -> bool:
        return var_label in self.candidate.var_assignments
