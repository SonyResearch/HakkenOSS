from __future__ import annotations

import textwrap
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Sequence

    from query_common.values.keywords import Operator

TClause = TypeVar("TClause", bound="Clause")


class Clause(ABC):
    def __repr__(self) -> str:
        return self.to_str()

    @abstractmethod
    def to_str(self, indent_length: int = 4) -> str:
        """Return a string representation of the Abstract Syntax Tree."""
        pass


class OperationClause(Clause, Generic[TClause]):
    def __init__(self, operator: Operator, operands: Sequence[TClause]):
        self.operator = operator
        self.operands = operands

    def to_str(self, indent_length: int = 4) -> str:
        tree = f"<{self.__class__.__name__}>\n"
        subtree = f"operator: {self.operator!r}\n"
        for arg in self.operands:
            subtree += "arg:\n"
            subtree += textwrap.indent(arg.to_str(), prefix=" " * indent_length) + "\n"
        tree += textwrap.indent(subtree, prefix=" " * indent_length)
        return tree
