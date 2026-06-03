from enum import StrEnum
from typing import Final

EXISTS_OPERATOR: Final[str] = "EXISTS"


class QueryingType(StrEnum):
    SIMPLE = "simple"


class ConditionType(StrEnum):
    LEAF = "leaf"
    AND = "and"
    OR = "or"
    NOT = "not"
