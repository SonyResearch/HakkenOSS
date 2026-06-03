from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

from query_common.entities.conditions.link import LinkCondition
from query_common.entities.conditions.logical import NegatedCondition

from complex_query.core.values.errors import SearchInputError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from query_common.entities.conditions.base import Condition
    from query_common.entities.variable import Variable


def get_execution_order(conditions: Sequence[Condition]) -> list[int]:
    """Given a list of conditions, return the order in which to execute them.
    The execution order is calculated based on the variables in each condition.
    It dictates in which order the variables should be substituted.

    :param conditions: List of link (s,r,o) conditions
    :return: List of conditions (identified by index) in order of execution.
    """
    for c in conditions:
        if not (
            isinstance(c, LinkCondition)
            or (isinstance(c, NegatedCondition) and isinstance(c.condition, LinkCondition))
        ):
            raise SearchInputError(
                "For now, only link conditions or negated link conditions are "
                f"supported. Received {c} instead."
            )
    # The execution order is calculated based on the variables in each condition:
    # it dictates in which order these variables will be substituted.
    condition_to_vars: dict[int, list[Variable]] = {
        c_i: c.variables() for c_i, c in enumerate(conditions)
    }
    execution_order: list[int] = []
    while len(execution_order) < len(conditions):
        # We sort the remaining conditions by how many variables they have that have not been
        # substituted yet (increasing order).
        order_by_least_vars: list[int] = sorted(
            range(len(condition_to_vars)), key=lambda idx: len(condition_to_vars[idx])
        )
        order_by_least_vars = [idx for idx in order_by_least_vars if idx not in execution_order]
        # The potential next conditions to execute are all the ones
        # with the least non-substituted variables.
        next_in_tie = [
            idx
            for idx in order_by_least_vars
            if len(condition_to_vars[idx]) == len(condition_to_vars[order_by_least_vars[0]])
        ]
        # Additional priorities
        type_priorities: list[type[Condition]] = [LinkCondition, NegatedCondition]
        next_in_tie_per_subtypes = (
            [idx for idx in next_in_tie if isinstance(conditions[idx], type_priority)]
            for type_priority in type_priorities
        )
        next_in_tie = list(itertools.chain(*next_in_tie_per_subtypes))
        next_in_order = next_in_tie.pop(0)
        execution_order.append(next_in_order)
        # Update the variables that have been already substituted.
        executed_vars = condition_to_vars[next_in_order]
        for c_i, variables in condition_to_vars.items():
            condition_to_vars[c_i] = [v for v in variables if v not in executed_vars]
        condition_to_vars = {
            c_i: [v for v in variables if v not in executed_vars]
            for c_i, variables in condition_to_vars.items()
        }
    return execution_order
