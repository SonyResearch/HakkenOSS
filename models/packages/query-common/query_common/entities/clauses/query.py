from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

from query_common.entities.clauses.clause import Clause

if TYPE_CHECKING:
    from collections.abc import Sequence

    from query_common.entities.clauses.formula import Formula
    from query_common.entities.clauses.term import DomainTerm, VariableTerm


class Query(Clause):
    def __init__(self, patterns: Sequence[Pattern], condition: Formula):
        self.patterns = patterns
        self.condition = condition

    def to_str(self, indent_length: int = 4) -> str:
        tree = f"<{self.__class__.__name__}>\n"
        subtree = "patterns:\n"
        for pattern in self.patterns:
            subtree += textwrap.indent(pattern.to_str(), prefix=" " * indent_length) + "\n"
        subtree += "condition:\n"
        subtree += textwrap.indent(self.condition.to_str(), prefix=" " * indent_length) + "\n"
        tree += textwrap.indent(subtree, prefix=" " * indent_length)
        return tree


class Pattern(Clause):
    def __init__(self, variable: VariableTerm, domain: DomainTerm):
        self.variable = variable
        self.domain = domain

    def to_str(self, indent_length: int = 4) -> str:
        tree = f"<{self.__class__.__name__}>\n"
        subtree = "variable:\n"
        subtree += textwrap.indent(self.variable.to_str(), prefix=" " * indent_length) + "\n"
        subtree += "domain:\n"
        subtree += textwrap.indent(self.domain.to_str(), prefix=" " * indent_length) + "\n"
        tree += textwrap.indent(subtree, prefix=" " * indent_length)
        return tree
