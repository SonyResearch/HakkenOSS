from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from query_common.entities.conditions.base import Condition
    from query_common.entities.variable import Variable, VarLabel


@dataclass
class GroundedQuery:
    variables: dict["VarLabel", "Variable"]
    condition: "Condition"

    def __str__(self) -> str:
        return (
            f"GroundedQuery(vars={[str(variable) for variable in self.variables]},"
            f" condition={self.condition!s}"
        )

    def __repr__(self) -> str:
        return str(self)
