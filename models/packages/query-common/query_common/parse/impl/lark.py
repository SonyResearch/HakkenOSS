from typing import Literal

import lark

from query_common.entities.clauses.formula import Atom, ConnectedFormula, Formula
from query_common.entities.clauses.query import Pattern, Query
from query_common.entities.clauses.term import (
    DomainTerm,
    FloatTerm,
    FunctionTerm,
    IntegerTerm,
    PropertyTerm,
    Term,
    VariableTerm,
)
from query_common.parse.base import Parser
from query_common.values.errors import ParsingLogicError
from query_common.values.keywords import (
    ArithmeticOperator,
    ComparisonOperator,
    FunctionSymbol,
    LogicalOperator,
    PredicateSymbol,
    PropertyDelimiterT,
    PropertySymbol,
)

# Patterns to be matched:
# 1. Alphanumeric strings not starting with a number, but not AND, NOT, OR
# 2. All alphanumeric strings encapsulated with single or double quotes
SYMBOL = (
    rf"(?!{LogicalOperator.AND.value}$)"
    rf"(?!{LogicalOperator.NOT.value}$)"
    rf"(?!{LogicalOperator.OR.value}$)"
    rf"[a-zA-Z_][a-zA-Z0-9_]*"
)
QUOTED_STRING = r"'[^'\n\r]+'|\"[^\"\n\r]+\""
# Must start with a letter (a-z, A-Z) or an underscore (_).
# Can be followed by any number of letters (a-z, A-Z), digits (0-9), or underscores (_).
# Is not "AND", "NOT", "OR"


class LarkParser(Parser):
    def __init__(self):
        super().__init__()
        grammar = rf"""
            start: query

            query: "?" patterns "WHERE" formula | formula

            patterns: pattern ("," pattern)*
            pattern: variable "in" domain


            formula: or_formula | and_formula | not_formula | delimited_formula | atom
            delimited_formula: "("formula")"
            atom: comparison_atom | predicate_atom
            not_formula: "{LogicalOperator.NOT.value}" formula
            and_formula: formula "{LogicalOperator.AND.value}" formula
            or_formula: formula "{LogicalOperator.OR.value}" formula
            comparison_atom: term comparator term
            comparator: {" | ".join(['"' + x.value + '"' for x in ComparisonOperator])}
            predicate_atom: predicate"("terms")"

            terms: term ("," term)*
            term: variable | function_term | property_term | infix_function_term
                | integer | float
            function_term: function"("terms")"
            property_term: term"."property
            infix_function_term: term infix_function term
            infix_function: {" | ".join(['"' + x.value + '"' for x in ArithmeticOperator])}


            predicate: SYMBOL
            function: SYMBOL
            variable: SYMBOL | QUOTED_STRING
            domain: SYMBOL | QUOTED_STRING
            property: SYMBOL
            integer: SIGNED_INT
            float: SIGNED_FLOAT

            SYMBOL: /{SYMBOL}/
            QUOTED_STRING: /{QUOTED_STRING}/
            %import common.WS
            %import common.SIGNED_FLOAT
            %import common.SIGNED_INT
            %ignore WS
        """
        self.parser = lark.Lark(grammar, keep_all_tokens=True)
        self.transformer = StringToClauseTransformer()

    def parse_query(self, text: str) -> Query:
        parse_result = self.parser.parse(text)
        return self.transformer.transform(parse_result)


def _unescape_string(value: str):
    if value[0] == value[-1] == '"' or value[0] == value[-1] == "'":
        value = value[1:-1]
    return value


class StringToClauseTransformer(lark.Transformer[lark.Token, Query]):
    """This transformer takes a string as input and converts it to a Query clause"""

    # Symbols
    def predicate(self, tok: tuple[str]) -> PredicateSymbol:
        return str(tok[0])

    def function(self, tok: tuple[str]) -> FunctionSymbol:
        return str(tok[0])

    def property(self, tok: tuple[str]) -> PropertySymbol:
        return str(tok[0])

    def variable(self, tok: tuple[str]) -> VariableTerm:
        return VariableTerm(value=_unescape_string(tok[0]))

    # Clauses
    def float(self, tok: tuple[str]) -> FloatTerm:
        return FloatTerm(value=float(tok[0]))

    def integer(self, tok: tuple[str]) -> IntegerTerm:
        return IntegerTerm(value=int(tok[0]))

    def domain(self, tok: tuple[str]) -> DomainTerm:
        return DomainTerm(value=_unescape_string(str(tok[0])))

    def infix_function(self, tok: tuple[str]) -> ArithmeticOperator:
        return ArithmeticOperator(str(tok[0]))

    def infix_function_term(self, tok: tuple[Term, FunctionSymbol, Term]) -> FunctionTerm:
        return FunctionTerm(function_name=tok[1], arguments=[tok[0], tok[2]])

    def property_term(self, tok: tuple[Term, PropertyDelimiterT, PropertySymbol]) -> PropertyTerm:
        return PropertyTerm(term=tok[0], property=tok[2])

    def function_term(
        self,
        tok: tuple[FunctionSymbol, Literal["("], list[Term], Literal[")"]],
    ) -> FunctionTerm:
        return FunctionTerm(function_name=tok[0], arguments=tok[2])

    def term(self, tok: tuple[Term]) -> Term:
        return tok[0]

    def predicate_atom(
        self,
        tok: tuple[PredicateSymbol, Literal["("], list[Term], Literal[")"]],
    ) -> Atom:
        return Atom(operator=tok[0], operands=tok[2])

    def comparator(self, tok: tuple[str]) -> ComparisonOperator:
        return ComparisonOperator(str(tok[0]))

    def comparison_atom(self, tok: tuple[Term, ComparisonOperator, Term]) -> Atom:
        return Atom(operator=tok[1], operands=[tok[0], tok[2]])

    def atom(self, tok: tuple[Atom]) -> Atom:
        return tok[0]

    def not_formula(self, tok: tuple[str, Formula]) -> ConnectedFormula:
        return ConnectedFormula(operator=LogicalOperator(str(tok[0])), operands=[tok[1]])

    def and_formula(self, tok: tuple[Formula, LogicalOperator, Formula]) -> ConnectedFormula:
        return ConnectedFormula(operator=LogicalOperator.AND, operands=[tok[0], tok[2]])

    def or_formula(self, tok: tuple[Formula, LogicalOperator, Formula]) -> ConnectedFormula:
        return ConnectedFormula(operator=LogicalOperator.OR, operands=[tok[0], tok[2]])

    def formula(self, tok: tuple[Formula]) -> Formula:
        return tok[0]

    def delimited_formula(self, tok: tuple[Literal["("], Formula, Literal[")"]]) -> Formula:
        return tok[1]

    def pattern(self, tok: tuple[VariableTerm, Literal["in"], DomainTerm]) -> Pattern:
        return Pattern(variable=tok[0], domain=tok[2])

    def query(
        self,
        tok: (tuple[Formula] | tuple[Literal["?"], list[Pattern], Literal["WHERE"], Formula]),
    ) -> Query:
        if len(tok) == 4:  # noqa: PLR2004
            return Query(patterns=tok[1], condition=tok[3])
        if len(tok) == 1:
            return Query(patterns=[], condition=tok[0])

        raise ParsingLogicError(
            "A query should be a formula (single token [formula]), or a pattern "
            "(four tokens ['?', pattern, 'WHERE', formula]). "
            "Parsed an incompatible number of tokens."
        )

    def patterns(self, tok: tuple[Pattern, ...]) -> tuple[Pattern]:
        return tok[0::2]  # type: ignore

    def terms(self, tok: tuple[Term, ...]) -> tuple[Term]:
        return tok[0::2]  # type: ignore

    def start(self, tok: tuple[Query]) -> Query:
        return tok[0]
