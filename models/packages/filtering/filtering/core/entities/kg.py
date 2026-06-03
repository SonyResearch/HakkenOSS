from enum import StrEnum
from typing import NamedTuple, TypeAlias

NodeId: TypeAlias = str
RelationType: TypeAlias = str
Year: TypeAlias = int


class EdgeDirection(StrEnum):
    IN = "in"
    OUT = "out"
    ALL = "all"


class YearRange(NamedTuple):
    start: Year
    end: Year
