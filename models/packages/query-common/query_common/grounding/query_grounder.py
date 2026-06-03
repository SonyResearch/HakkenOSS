from typing import TYPE_CHECKING

from query_common.entities.clauses.formula import Atom, ConnectedFormula, Formula
from query_common.entities.clauses.term import VariableTerm
from query_common.entities.conditions.link import LinkCondition
from query_common.entities.conditions.logical import (
    ConjunctiveCondition,
    DisjunctiveCondition,
    NegatedCondition,
)
from query_common.entities.kg.concept import Concept
from query_common.entities.kg.relation import Relation
from query_common.values.errors import GroundingInputError, GroundingLogicError
from query_common.values.keywords import LinkPredicate, LogicalOperator

if TYPE_CHECKING:
    from query_common.entities.conditions.base import AtomicCondition, Condition
    from query_common.entities.variable import Variable, VarLabel


class QueryGrounder:
    def __init__(self, variables: dict["VarLabel", "Variable"]) -> None:
        self.new_condition_id = 0
        self.variables = variables

    def get_new_condition_id(self) -> int:
        id_ = self.new_condition_id
        self.new_condition_id += 1
        return id_

    def convert_formula_to_condition(self, formula: Formula) -> "Condition":
        if not formula.is_dnf():
            raise GroundingInputError("Only DNF conditions are supported.")
        if formula.is_literal():
            condition = self.convert_literal_to_condition(formula)
        else:
            if not isinstance(formula, ConnectedFormula):
                raise GroundingLogicError(
                    "Expected connected formula for DNF that is not a literal."
                )
            if formula.is_conjunctive_clause():
                # AND with operands that are all literals
                condition = ConjunctiveCondition(
                    self.get_new_condition_id(),
                    [self.convert_formula_to_condition(lit) for lit in formula.operands],
                )
            else:
                # Disjunction of CNFs
                conditions = []
                for lit in formula.operands:
                    conditions.append(self.convert_formula_to_condition(lit))
                condition = DisjunctiveCondition(self.get_new_condition_id(), conditions)
        return condition

    def convert_literal_to_condition(self, formula: Formula) -> "Condition":
        # TODO: Update the code to have an explicit type for clauses.Literal
        #  This will make logic and type checking much easier.
        if formula.is_positive_literal():
            if not isinstance(formula, Atom):
                raise GroundingLogicError("Expected Atom for a positive literal.")
            return self.convert_atom_to_condition(formula)

        if not isinstance(formula, ConnectedFormula):
            raise GroundingLogicError("Expected connected formula for a negative literal")
        atom = formula.operands[0]
        if not (isinstance(atom, Atom) and formula.operator == LogicalOperator.NOT):
            raise GroundingLogicError(
                "Expected NOT operator and Atom operand for a negative literal."
            )
        return NegatedCondition(self.get_new_condition_id(), self.convert_atom_to_condition(atom))

    def convert_atom_to_condition(self, atom: Atom) -> "AtomicCondition":
        """For now only supports simply link atoms of the form P(S,R,O)"""
        if not (atom.operator == LinkPredicate and len(atom.operands) == 3):  # noqa: PLR2004
            raise GroundingInputError(f"Only supports {LinkPredicate}(S,R,O) links for now.")
        triple_relation_term = atom.operands[1]
        triple_subject_term = atom.operands[0]
        triple_object_term = atom.operands[2]
        if not isinstance(triple_subject_term, VariableTerm):
            raise GroundingInputError(
                "For now, only constant/variable terms are supported. "
                f"Received {type(triple_subject_term)}."
            )
        if not isinstance(triple_relation_term, VariableTerm):
            raise GroundingInputError(
                "For now, only constant/variable terms are supported. "
                f"Received {type(triple_relation_term)}."
            )
        if not isinstance(triple_object_term, VariableTerm):
            raise GroundingInputError(
                "For now, only constant/variable terms are supported. "
                f"Received {type(triple_object_term)}."
            )
        triple_relation = Relation(identifier=str(triple_relation_term.value))
        triple_subject: Variable | Concept = (
            self.variables[triple_subject_term.value]
            if triple_subject_term.value in self.variables
            else Concept(identifier=str(triple_subject_term.value))
        )
        triple_object: Variable | Concept = (
            self.variables[triple_object_term.value]
            if triple_object_term.value in self.variables
            else Concept(identifier=str(triple_object_term.value))
        )

        return LinkCondition(
            id_=self.get_new_condition_id(),
            subject=triple_subject,
            relation=triple_relation,
            object_=triple_object,
        )
