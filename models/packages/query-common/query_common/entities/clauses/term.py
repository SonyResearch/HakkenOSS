from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING, Any

from query_common.entities.clauses.clause import Clause, OperationClause
from query_common.values.errors import ParsingLogicError

if TYPE_CHECKING:
    from query_common.values.keywords import FunctionSymbol, PropertySymbol


class Term(Clause):
    pass


class FunctionTerm(OperationClause[Term]):
    def __init__(self, function_name: FunctionSymbol, arguments: list[Term]):
        super().__init__(operator=function_name, operands=arguments)
        self.operator: FunctionSymbol = function_name


class PropertyTerm(Term):
    def __init__(self, term: Term, property: PropertySymbol):
        self.term = term
        self.property = property

    def to_str(self, indent_length: int = 4) -> str:
        tree = f"<{self.__class__.__name__}>\n"
        subtree = "term:\n"
        subtree += textwrap.indent(self.term.to_str(), prefix=" " * indent_length)
        subtree += "\nproperty:\n"
        subtree += textwrap.indent(self.property, prefix=" " * indent_length)
        tree += textwrap.indent(subtree, prefix=" " * indent_length)
        return tree


class ValueTerm(Term):
    def __init__(self, value: Any):
        self.value = value

    def to_str(self, indent_length: int = 4) -> str:  # noqa: ARG002
        tree = f"<{self.__class__.__name__}>\n"
        tree += f"value: {self.value}"
        return tree


class VariableTerm(ValueTerm):
    def __init__(self, value: str):
        if not isinstance(value, str):
            raise ParsingLogicError(f"Expected a string but got {type(value)}")
        super().__init__(value)


class DomainTerm(ValueTerm):
    def __init__(self, value: str):
        if not isinstance(value, str):
            raise ParsingLogicError(f"Expected a string but got {type(value)}")
        super().__init__(value)


class IntegerTerm(ValueTerm):
    def __init__(self, value: int):
        if not isinstance(value, int):
            raise ParsingLogicError(f"Expected an integer but got {type(value)}")
        super().__init__(value)


class FloatTerm(ValueTerm):
    def __init__(self, value: float):
        if not isinstance(value, float):
            raise ParsingLogicError(f"Expected a float but got {type(value)}")
        super().__init__(value)
