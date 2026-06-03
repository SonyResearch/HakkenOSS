from abc import abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

from query_common.entities.clauses.clause import Clause, OperationClause
from query_common.entities.clauses.term import Term
from query_common.values.keywords import LogicalOperator, PredicateSymbol

if TYPE_CHECKING:
    from collections.abc import Sequence


class Formula(Clause):
    @abstractmethod
    def is_cnf(self) -> bool:
        """A formula is in conjunctive normal form (CNF) if it is a conjunction of
        one or more disjunctive clauses (disjunctions of literals)."""
        pass

    @abstractmethod
    def is_dnf(self) -> bool:
        """A formula is in disjunctive normal form (DNF) if it is a disjunction of
        one or more conjunctive clauses (conjunctions of literals).
        """
        pass

    @abstractmethod
    def is_conjunctive_clause(self) -> bool:
        """A conjunctive clause is a conjunction of literals"""
        pass

    @abstractmethod
    def is_disjunctive_clause(self) -> bool:
        """A disjunctive clause is a disjunction of literals"""
        pass

    @abstractmethod
    def is_literal(self) -> bool:
        """A literal is an atom or its negation.
        An atom is any predicate applied to any set of terms."""
        pass

    @abstractmethod
    def is_positive_literal(self) -> bool:
        """A positive literal is an atom.
        An atom is any predicate applied to any set of terms."""
        pass

    @abstractmethod
    def is_negative_literal(self) -> bool:
        """A positive literal is the negation of an atom.
        An atom is any predicate applied to any set of terms."""
        pass


class ConnectedFormula(OperationClause[Formula], Formula):
    def __init__(self, operator: LogicalOperator, operands: "Sequence[Formula]"):
        super().__init__(operator, operands)
        self.operator: LogicalOperator = operator
        if self.operator == LogicalOperator.NOT and len(operands) != 1:
            raise ValueError(
                f"`LogicalOperator.NOT` should have a unique argument. Found {len(operands)}."
            )

    def is_cnf(self) -> bool:
        is_conjunction_of_cnf = self.operator == LogicalOperator.AND and all(
            op.is_cnf() for op in self.operands
        )
        is_conjunction_of_disjunctive_clause = self.operator == LogicalOperator.AND and all(
            op.is_disjunctive_clause() for op in self.operands
        )
        return (
            self.is_disjunctive_clause()
            or is_conjunction_of_cnf
            or is_conjunction_of_disjunctive_clause
        )

    def is_dnf(self) -> bool:
        is_disjunction_of_dnf = self.operator == LogicalOperator.OR and all(
            op.is_dnf() for op in self.operands
        )
        is_disjunction_of_conjunctive_clause = self.operator == LogicalOperator.OR and all(
            op.is_conjunctive_clause() for op in self.operands
        )
        return (
            self.is_conjunctive_clause()
            or is_disjunction_of_dnf
            or is_disjunction_of_conjunctive_clause
        )

    def is_conjunctive_clause(self) -> bool:
        ops: Sequence[Formula] = self.operands
        return self.is_literal() or (
            self.operator == LogicalOperator.AND and all(op.is_conjunctive_clause() for op in ops)
        )

    def is_disjunctive_clause(self) -> bool:
        ops: Sequence[Formula] = self.operands
        return self.is_literal() or (
            self.operator == LogicalOperator.OR and all(op.is_disjunctive_clause() for op in ops)
        )

    def is_literal(self) -> bool:
        return self.operator == LogicalOperator.NOT and isinstance(self.operands[0], Atom)

    def is_positive_literal(self) -> bool:
        return False

    def is_negative_literal(self) -> bool:
        return self.is_literal()


class Atom(OperationClause[Term], Formula):
    def __init__(self, operator: PredicateSymbol, operands: "Sequence[Term]"):
        super().__init__(operator, operands)
        self.operator: PredicateSymbol = operator

    def is_cnf(self) -> bool:
        return True

    def is_dnf(self) -> bool:
        return True

    def is_conjunctive_clause(self) -> bool:
        return True

    def is_disjunctive_clause(self) -> bool:
        return True

    def is_literal(self) -> bool:
        return True

    def is_positive_literal(self) -> bool:
        return True

    def is_negative_literal(self) -> bool:
        return False
