from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, cast

from pydantic import BaseModel, field_validator, model_validator
from query_common.entities.clauses.term import Term, ValueTerm
from query_common.entities.kg.identifier import DomainIdentifier
from query_common.grounding.query_grounder import Atom, ConnectedFormula, Formula
from query_common.values.keywords import LogicalOperator

from simple_query.query.values.errors import PredicateError, QueryInputConversionError
from simple_query.query.values.types import EXISTS_OPERATOR, ConditionType

if TYPE_CHECKING:
    from typing import Self

    from query_common.entities.clauses.query import Query

PredicateT = TypeVar("PredicateT", bound="Predicate")


class Argument(BaseModel):
    """
    Defines an argument (either subject, relation, or object) of a `Predicate`.
    """

    value: str
    """A value, which can be either a node ID or relation type (when `not is_variable`)
    or a variable (e.g. `X`, when `is_variable`)"""
    is_variable: bool = False
    """Specifies whether the argument is variable."""


class Predicate(BaseModel):
    """
    Base class for predicates.
    """

    name: str
    subject: Argument
    relation: Argument
    object: Argument


class TargetPredicate(Predicate):
    """
    Class for target predicates, i.e. a triple that doesn't exist in the KG.

    The validator checks `name` value, to ensure the name is not the reserved `EXISTS_OPERATOR`.
    """

    @field_validator("name", mode="before")
    @classmethod
    def check_name(cls, name: str) -> str:
        if name == EXISTS_OPERATOR:
            raise PredicateError(f"Name of target predicate should not be `{EXISTS_OPERATOR}`")
        return name


class ConditionPredicate(Predicate):
    """
    Class for condition predicates, i.e. a triple that exists in the KG.

    The `name` field should not be explicitly set, and the validator checks whether it has
    different value from `EXISTS_OPERATOR`.
    """

    name: str = EXISTS_OPERATOR

    @field_validator("name", mode="before")
    @classmethod
    def check_name(cls, name: str) -> str:
        if name != EXISTS_OPERATOR:
            raise PredicateError(f"Name of condition predicate should be `{EXISTS_OPERATOR}`")
        return name


class ConditionNode(BaseModel):
    """
    Node of a condition tree.

    If `type` is `ConditionType.LEAF`, it indicates that the node is a leaf node,
    and should be associated with a predicate.
    If not, it indicates that the node is a non-leaf node, where it should have children
    instead.
    """

    type: ConditionType
    predicate: ConditionPredicate | None = None
    children: list[ConditionNode] = []

    @model_validator(mode="after")
    def check_field_values(self) -> Self:
        if self.type == ConditionType.LEAF:
            if self.predicate is None:
                raise ValueError("`predicate` should be given for a leaf condition node")
            if self.children:
                raise ValueError("`children` should be empty for a leaf condition node")
        else:
            if self.predicate:
                raise ValueError("`predicate` should be `None` for a non-leaf condition node")
            if self.type == ConditionType.NOT and len(self.children) != 1:
                raise ValueError("NOT condition node should have 1 child")
            if self.type in (ConditionType.AND, ConditionType.OR) and len(self.children) < 2:  # noqa: PLR2004
                raise ValueError("AND or OR condition node should have at least 2 children")
        return self


class QueryInput(BaseModel):
    """
    Defines the input to querying model.
    """

    target_predicate: TargetPredicate
    variable_name: str
    variable_domain_identifier: DomainIdentifier
    condition: ConditionNode | None


def _extract_target_predicate_and_condition_from_formula(
    formula: Formula, variable_name: str
) -> tuple[TargetPredicate, ConditionNode | None]:
    if isinstance(formula, ConnectedFormula):
        if formula.operator != LogicalOperator.AND:
            raise QueryInputConversionError(
                "the formula to extract target and condition should be a `ConnectedFormula` of AND"
            )

        operands = formula.operands
        if len(operands) != 2:  # noqa: PLR2004
            raise QueryInputConversionError(
                f"the formula should have 2 operands at the top level, but got {formula}"
            )

        if isinstance(operands[0], Atom):
            target_formula = operands[0]
            condition_formula = operands[1]
        elif isinstance(operands[1], Atom):
            target_formula = operands[1]
            condition_formula = operands[0]
        else:
            raise QueryInputConversionError(
                f"the formula does not have an atomic formula at the top level, but got {formula}"
            )
    elif isinstance(formula, Atom):
        target_formula = formula
        condition_formula = None

    def convert_to_argument(value_term: Term) -> Argument:
        if not isinstance(value_term, ValueTerm):
            raise QueryInputConversionError(f"the term is not a value term: {value_term}")
        value = str(value_term.value)
        return Argument(value=value, is_variable=value == variable_name)

    def convert_to_predicate(atomic_formula: Atom, predicate_class: type[PredicateT]) -> PredicateT:
        return predicate_class(
            name=cast("str", atomic_formula.operator),
            subject=convert_to_argument(atomic_formula.operands[0]),
            relation=convert_to_argument(atomic_formula.operands[1]),
            object=convert_to_argument(atomic_formula.operands[2]),
        )

    target_predicate = convert_to_predicate(
        cast("Atom", target_formula), predicate_class=TargetPredicate
    )

    def convert_condition_formula_to_condition(
        condition_formula: Formula,
    ) -> ConditionNode:
        if isinstance(condition_formula, ConnectedFormula):
            operator = condition_formula.operator
            operands = condition_formula.operands

            if operator == LogicalOperator.AND:
                condition_type = ConditionType.AND
            elif operator == LogicalOperator.OR:
                condition_type = ConditionType.OR
            elif operator == LogicalOperator.NOT:
                condition_type = ConditionType.NOT
            else:
                raise QueryInputConversionError(f"unknown operator: {operator}")

            return ConditionNode(
                type=condition_type,
                predicate=None,
                children=[convert_condition_formula_to_condition(operand) for operand in operands],
            )

        if isinstance(condition_formula, Atom):
            condition_predicate = convert_to_predicate(
                condition_formula, predicate_class=ConditionPredicate
            )
            return ConditionNode(type=ConditionType.LEAF, predicate=condition_predicate)

        raise QueryInputConversionError(
            f"unknown condition formula type: {type(condition_formula)}"
        )

    if condition_formula is not None:
        condition = convert_condition_formula_to_condition(condition_formula)
    else:
        condition = None

    return target_predicate, condition


def convert_to_query_input(
    query: Query,
    variable_name: str | None = None,
    variable_domain_identifier: DomainIdentifier | None = None,
) -> QueryInput:
    """
    Converts a parsed query object (from `query-common`) into the query input used in
    `Querying` model in this package.
    """
    patterns = query.patterns
    formula = query.condition

    for pattern in patterns:
        if variable_name is not None or variable_domain_identifier is not None:
            raise QueryInputConversionError(
                f"conversion to simple query input failed due to multiple domain information "
                f"for variable {pattern.variable}"
            )

        variable_name = str(pattern.variable.value)
        domain_str = str(pattern.domain.value)
        variable_domain_identifier = domain_str

    if variable_name is None or variable_domain_identifier is None:
        raise QueryInputConversionError(
            f"could not find variable name and domain from query {query}"
        )

    target_predicate, condition = _extract_target_predicate_and_condition_from_formula(
        formula=formula, variable_name=variable_name
    )

    return QueryInput(
        target_predicate=target_predicate,
        variable_name=variable_name,
        variable_domain_identifier=variable_domain_identifier,
        condition=condition,
    )
