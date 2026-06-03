from enum import StrEnum
from typing import Annotated, Literal


class LogicalOperator(StrEnum):
    AND = "AND"
    OR = "OR"
    NOT = "NOT"


class ComparisonOperator(StrEnum):
    EQUALS = "="
    NOT_EQUALS = "!="
    GREATER_THAN_OR_EQUALS = ">="
    GREATER_THAN = ">"
    LESS_THAN_OR_EQUALS = "<="
    LESS_THAN = "<"


class ArithmeticOperator(StrEnum):
    ADD = "+"
    SUBTRACT = "-"
    MULTIPLY = "*"
    DIVIDE = "/"


PredicateSymbol = Annotated[str, "Predicate Symbol"] | ComparisonOperator
FunctionSymbol = Annotated[str, "Function Symbol"] | ArithmeticOperator
PropertySymbol = Annotated[str, "Property Symbol"]
Symbol = PredicateSymbol | FunctionSymbol | PropertySymbol

Operator = LogicalOperator | PredicateSymbol | FunctionSymbol

PropertyDelimiter = "."
PropertyDelimiterT = Literal["."]
LinkPredicate = "P"
