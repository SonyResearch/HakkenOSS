from enum import StrEnum
from typing import NoReturn


class NotGiven:
    def __bool__(self) -> NoReturn:
        raise NotImplementedError("Boolean conversion is not supported for `NotGiven` objects.")

    def __repr__(self) -> str:
        return "NotGiven"


NOT_GIVEN = NotGiven()


class SearchType(StrEnum):
    BEAM = "beam"


class KGType(StrEnum):
    NETWORKX = "networkx"
    NEO4J = "neo4j"
    CACHED = "cached"


class KGLedgerType(StrEnum):
    HDF5 = "hdf5"
    IN_MEMORY = "in_memory"
    SQLITE = "sqlite"


class ScoreLedgerType(StrEnum):
    IN_MEMORY = "in_memory"


class ScoreAggregatorType(StrEnum):
    PRODUCT = "product"
    MINIMUM = "minimum"


class LinkPredictorType(StrEnum):
    RANDOM = "random"
    API_BASED = "api_based"
