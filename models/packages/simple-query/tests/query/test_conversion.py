from typing import TYPE_CHECKING

import pytest
from query_common.parse.impl.lark import LarkParser

from simple_query.query.entities.inputs import convert_to_query_input
from simple_query.query.values.errors import QueryInputConversionError
from simple_query.query.values.types import EXISTS_OPERATOR, ConditionType

if TYPE_CHECKING:
    from query_common.parse.base import Parser


@pytest.fixture
def parser() -> "Parser":
    return LarkParser()


def test_conversion_no_condition(parser):
    query_str = "P(X, r1, n3)"
    parsed_query = parser.parse_query(query_str)
    query_input = convert_to_query_input(
        query=parsed_query, variable_name="X", variable_domain_identifier="GENE"
    )

    assert query_input.target_predicate.name == "P"
    assert query_input.target_predicate.subject.value == "X"
    assert query_input.target_predicate.subject.is_variable
    assert query_input.target_predicate.relation.value == "r1"
    assert not query_input.target_predicate.relation.is_variable
    assert query_input.target_predicate.object.value == "n3"
    assert not query_input.target_predicate.object.is_variable

    assert query_input.condition is None

    assert query_input.variable_name == "X"
    assert query_input.variable_domain_identifier == "GENE"


def test_conversion_single_condition(parser):
    query_str = "P(X, r1, n3) AND EXISTS(n4, r2, X)"
    parsed_query = parser.parse_query(query_str)
    query_input = convert_to_query_input(
        query=parsed_query, variable_name="X", variable_domain_identifier="GENE"
    )

    assert query_input.target_predicate.name == "P"
    assert query_input.target_predicate.subject.value == "X"
    assert query_input.target_predicate.subject.is_variable
    assert query_input.target_predicate.relation.value == "r1"
    assert not query_input.target_predicate.relation.is_variable
    assert query_input.target_predicate.object.value == "n3"
    assert not query_input.target_predicate.object.is_variable

    assert query_input.condition.type == ConditionType.LEAF
    assert not query_input.condition.children
    assert query_input.condition.predicate.name == EXISTS_OPERATOR
    assert query_input.condition.predicate.subject.value == "n4"
    assert not query_input.condition.predicate.subject.is_variable
    assert query_input.condition.predicate.relation.value == "r2"
    assert not query_input.condition.predicate.relation.is_variable
    assert query_input.condition.predicate.object.value == "X"
    assert query_input.condition.predicate.object.is_variable

    assert query_input.variable_name == "X"
    assert query_input.variable_domain_identifier == "GENE"


def test_conversion_nested_condition(parser):
    query_str = (
        "Q(Y, r1, n3) "
        "AND ("
        "    EXISTS(n4, r2, Y) "
        "    AND NOT (EXISTS(Y, r1, n5) OR NOT EXISTS(n2, r3, Y))"
        "    OR EXISTS(Y, r2, n4)"
        ")"
    )
    parsed_query = parser.parse_query(query_str)
    query_input = convert_to_query_input(
        query=parsed_query, variable_name="Y", variable_domain_identifier="CHEMICAL"
    )

    assert query_input.target_predicate.name == "Q"
    assert query_input.target_predicate.subject.value == "Y"
    assert query_input.target_predicate.subject.is_variable
    assert query_input.target_predicate.relation.value == "r1"
    assert not query_input.target_predicate.relation.is_variable
    assert query_input.target_predicate.object.value == "n3"
    assert not query_input.target_predicate.object.is_variable

    assert query_input.variable_name == "Y"
    assert query_input.variable_domain_identifier == "CHEMICAL"

    condition_root = query_input.condition
    assert condition_root.type == ConditionType.OR
    assert len(condition_root.children) == 2

    condition_1 = condition_root.children[0]
    assert condition_1.type == ConditionType.AND
    assert len(condition_1.children) == 2

    condition_1_1 = condition_1.children[0]  # EXISTS(n4, r2, Y)
    assert condition_1_1.type == ConditionType.LEAF
    assert condition_1_1.predicate.subject.value == "n4"
    assert condition_1_1.predicate.relation.value == "r2"
    assert condition_1_1.predicate.object.value == "Y"

    condition_1_2 = condition_1.children[1]  # NOT (EXISTS(Y, r1, n5) OR NOT EXISTS(n2, r3, Y))
    assert condition_1_2.type == ConditionType.NOT
    assert len(condition_1_2.children) == 1
    assert condition_1_2.children[0].type == ConditionType.OR
    assert condition_1_2.children[0].children[0].type == ConditionType.LEAF
    assert condition_1_2.children[0].children[0].predicate.subject.value == "Y"
    assert condition_1_2.children[0].children[1].type == ConditionType.NOT
    assert condition_1_2.children[0].children[1].children[0].type == ConditionType.LEAF
    assert condition_1_2.children[0].children[1].children[0].predicate.subject.value == "n2"

    condition_2 = condition_root.children[1]  # EXISTS(Y, r2, n4)
    assert condition_2.type == ConditionType.LEAF
    assert condition_2.predicate.subject.value == "Y"
    assert condition_2.predicate.relation.value == "r2"
    assert condition_2.predicate.object.value == "n4"


def test_conversion_with_condition_in_query(parser):
    query_str = "P(X, r1, n3) AND EXISTS(n4, r2, X)"
    query_str_with_condition = "? X in 'GENE' WHERE P(X, r1, n3) AND EXISTS(n4, r2, X)"

    parsed_query = parser.parse_query(query_str)
    parsed_query_with_condition = parser.parse_query(query_str_with_condition)

    query_input = convert_to_query_input(
        parsed_query, variable_name="X", variable_domain_identifier="GENE"
    )
    query_input_with_condition = convert_to_query_input(parsed_query_with_condition)
    assert query_input == query_input_with_condition

    with pytest.raises(QueryInputConversionError):
        convert_to_query_input(
            parsed_query_with_condition, variable_name="X", variable_domain_identifier="GENE"
        )
