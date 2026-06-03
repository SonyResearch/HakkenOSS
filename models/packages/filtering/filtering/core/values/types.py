from enum import StrEnum


class KnowledgeGraphType(StrEnum):
    NETWORKX = "networkx"
    NEO4J = "neo4j"


class NodeFilteringType(StrEnum):
    ENTROPY = "entropy"
    RECENCY = "recency"
    RANDOM = "random"


class TripleFilteringType(StrEnum):
    RANDOM = "random"
