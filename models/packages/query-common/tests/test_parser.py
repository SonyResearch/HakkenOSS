from typing import TYPE_CHECKING

import pytest
from lark import LexError

from query_common.parse.impl.lark import LarkParser
from query_common.values.keywords import ArithmeticOperator, ComparisonOperator, LogicalOperator

if TYPE_CHECKING:
    from query_common.parse.base import Parser


@pytest.fixture(params=[LarkParser])
def parser(request):
    parser: Parser = request.param()
    return parser


def test_parse_query_operator_precedence(parser):
    q1 = parser.parse_query("P(x) AND NOT Q(x)")  # P(x) AND (NOT Q(x))
    assert q1.condition.operator == LogicalOperator.AND
    assert q1.condition.operands[0].operator == "P"
    assert q1.condition.operands[1].operator == LogicalOperator.NOT
    assert q1.condition.operands[1].operands[0].operator == "Q"

    q2 = parser.parse_query("P(x) AND Q(x) OR R(x)")  # (P(x) AND Q(x)) OR R(x)
    assert q2.condition.operator == LogicalOperator.OR
    assert q2.condition.operands[0].operator == LogicalOperator.AND
    assert q2.condition.operands[0].operands[0].operator == "P"
    assert q2.condition.operands[0].operands[1].operator == "Q"
    assert q2.condition.operands[1].operator == "R"

    q3 = parser.parse_query("P(x) OR Q(x) AND R(x)")  # P(x) OR (Q(x) AND R(x))
    assert q3.condition.operator == LogicalOperator.OR
    assert q3.condition.operands[0].operator == "P"
    assert q3.condition.operands[1].operator == LogicalOperator.AND
    assert q3.condition.operands[1].operands[0].operator == "Q"
    assert q3.condition.operands[1].operands[1].operator == "R"

    q4 = parser.parse_query("NOT P(x) OR Q(x) AND R(x)")  # (NOT P(x)) OR (Q(x) AND R(x))
    assert q4.condition.operator == LogicalOperator.OR
    assert q4.condition.operands[0].operator == LogicalOperator.NOT
    assert q4.condition.operands[0].operands[0].operator == "P"
    assert q4.condition.operands[1].operator == LogicalOperator.AND
    assert q4.condition.operands[1].operands[0].operator == "Q"
    assert q4.condition.operands[1].operands[1].operator == "R"

    q5 = parser.parse_query("P(x) AND Q(x) OR R(x) AND S(x)")  # (P(x) AND Q(x)) OR (R(x) AND S(x))
    assert q5.condition.operator == LogicalOperator.OR
    assert q5.condition.operands[0].operator == LogicalOperator.AND
    assert q5.condition.operands[0].operands[0].operator == "P"
    assert q5.condition.operands[0].operands[1].operator == "Q"
    assert q5.condition.operands[1].operator == LogicalOperator.AND
    assert q5.condition.operands[1].operands[0].operator == "R"
    assert q5.condition.operands[1].operands[1].operator == "S"

    q6 = parser.parse_query("P(x) OR Q(x) AND R(x) OR S(x)")  # (P(x) OR ((Q(x) AND R(x))) OR S(x)
    assert q6.condition.operator == LogicalOperator.OR
    assert q6.condition.operands[0].operator == LogicalOperator.OR
    assert q6.condition.operands[0].operands[0].operator == "P"
    assert q6.condition.operands[0].operands[1].operator == LogicalOperator.AND
    assert q6.condition.operands[0].operands[1].operands[0].operator == "Q"
    assert q6.condition.operands[0].operands[1].operands[1].operator == "R"
    assert q6.condition.operands[1].operator == "S"

    q7 = parser.parse_query("(P(x) OR Q(x)) AND (R(x) OR S(x))")
    assert q7.condition.operator == LogicalOperator.AND
    assert q7.condition.operands[0].operator == LogicalOperator.OR
    assert q7.condition.operands[0].operands[0].operator == "P"
    assert q7.condition.operands[0].operands[1].operator == "Q"
    assert q7.condition.operands[1].operator == LogicalOperator.OR
    assert q7.condition.operands[1].operands[0].operator == "R"
    assert q7.condition.operands[1].operands[1].operator == "S"


def test_parse_query_with_atomic_condition(parser):
    q1 = parser.parse_query("? x in X WHERE P(x)")
    assert q1.patterns[0].variable.value == "x"
    assert q1.patterns[0].domain.value == "X"
    assert q1.condition.operator == "P"
    assert q1.condition.operands[0].value == "x"

    q2 = parser.parse_query("? x in Drugs WHERE Treats(x, cancer)")
    assert q2.patterns[0].variable.value == "x"
    assert q2.patterns[0].domain.value == "Drugs"
    assert q2.condition.operator == "Treats"
    assert q2.condition.operands[0].value == "x"
    assert q2.condition.operands[1].value == "cancer"

    with pytest.raises(LexError):
        # Predicates with no arguments are NOT allowed
        parser.parse_query("? x in X WHERE P()")


def test_parse_query_with_complex_condition(parser):
    q1 = parser.parse_query("? x in X WHERE P(x) AND R(x,y)")
    assert q1.patterns[0].variable.value == "x"
    assert q1.patterns[0].domain.value == "X"
    assert q1.condition.operator == LogicalOperator.AND
    assert q1.condition.operands[0].operator == "P"
    assert q1.condition.operands[1].operator == "R"

    q2 = parser.parse_query("? x in X WHERE (NOT P(x)) AND R(x,y)")
    assert q2.patterns[0].variable.value == "x"
    assert q2.patterns[0].domain.value == "X"
    assert q2.condition.operator == LogicalOperator.AND
    assert q2.condition.operands[0].operator == LogicalOperator.NOT
    assert q2.condition.operands[0].operands[0].operator == "P"
    assert q2.condition.operands[1].operator == "R"

    q3 = parser.parse_query("? x in X WHERE NOT (P(x) AND R(x,y))")
    assert q3.patterns[0].variable.value == "x"
    assert q3.patterns[0].domain.value == "X"
    assert q3.condition.operator == LogicalOperator.NOT
    assert q3.condition.operands[0].operator == LogicalOperator.AND
    assert q3.condition.operands[0].operands[0].operator == "P"
    assert q3.condition.operands[0].operands[1].operator == "R"


def test_parse_query_with_multiple_patterns(parser):
    q = parser.parse_query("? x in X, y in Y WHERE P(x) AND R(x,y)")
    assert q.patterns[0].variable.value == "x"
    assert q.patterns[0].domain.value == "X"
    assert q.patterns[1].variable.value == "y"
    assert q.patterns[1].domain.value == "Y"
    assert q.condition.operator == LogicalOperator.AND
    assert q.condition.operands[0].operator == "P"
    assert q.condition.operands[1].operator == "R"


def test_parse_query_with_quoted_string(parser):
    q = parser.parse_query("? x in 'X+Y' WHERE P('x y z', \"1\", 'x, y, z')")
    assert q.patterns[0].variable.value == "x"
    assert q.patterns[0].domain.value == "X+Y"
    assert q.condition.operator == "P"
    assert q.condition.operands[0].value == "x y z"
    assert q.condition.operands[1].value == "1"
    assert q.condition.operands[2].value == "x, y, z"


@pytest.mark.parametrize("operator", [op.value for op in ComparisonOperator])
def test_parse_query_with_comparison_condition(parser, operator):
    parser.parse_query(f"? x in X WHERE x.property {operator} 42")


@pytest.mark.parametrize("operator", [op.value for op in ArithmeticOperator])
def test_parse_query_with_arithmetic_operator(parser, operator):
    parser.parse_query(f"? x in X WHERE x {operator} 42 != 3.14")


def test_parse_cnf(parser):
    assert parser.parse_query("a = 4").condition.is_cnf()
    assert parser.parse_query("P(x)").condition.is_cnf()
    assert parser.parse_query("NOT P(x)").condition.is_cnf()
    assert parser.parse_query("P(x) AND Q(x)").condition.is_cnf()
    assert parser.parse_query("P(x) OR Q(x)").condition.is_cnf()
    assert not parser.parse_query("NOT (P(x) AND Q(x))").condition.is_cnf()
    assert parser.parse_query("NOT P(x) AND Q(x)").condition.is_cnf()
    assert parser.parse_query("NOT P(x) OR Q(x)").condition.is_cnf()
    assert parser.parse_query("(NOT P(x) OR Q(x)) AND R(y)").condition.is_cnf()


def test_parse_dnf(parser):
    assert parser.parse_query("a = 4").condition.is_dnf()
    assert parser.parse_query("P(x)").condition.is_dnf()
    assert parser.parse_query("NOT P(x)").condition.is_dnf()
    assert parser.parse_query("P(x) OR Q(x)").condition.is_dnf()
    assert parser.parse_query("P(x) AND Q(x)").condition.is_dnf()
    assert not parser.parse_query("NOT (P(x) OR Q(x))").condition.is_dnf()
    assert parser.parse_query("NOT P(x) OR Q(x)").condition.is_dnf()
    assert parser.parse_query("NOT P(x) AND Q(x)").condition.is_dnf()
    assert parser.parse_query("(NOT P(x) AND Q(x)) OR R(y)").condition.is_dnf()


def test_triples(parser):
    parsed = parser.parse_query('P(X, "1234", "5678") AND NOT Q(abcd, efgh, "X")')
    assert parsed.condition.operands[0].operator == "P"
    assert parsed.condition.operands[0].operands[0].value == "X"
    assert parsed.condition.operands[0].operands[1].value == "1234"
    assert parsed.condition.operands[0].operands[2].value == "5678"
    assert parsed.condition.operands[1].operator == LogicalOperator.NOT
    assert parsed.condition.operands[1].operands[0].operator == "Q"
    assert parsed.condition.operands[1].operands[0].operands[0].value == "abcd"
    assert parsed.condition.operands[1].operands[0].operands[1].value == "efgh"
    assert parsed.condition.operands[1].operands[0].operands[2].value == "X"
