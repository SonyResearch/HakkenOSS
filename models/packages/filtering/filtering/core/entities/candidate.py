from pydantic import BaseModel
from typing_extensions import TypedDict

from filtering.core.entities.kg import NodeId


class SymbolMapping(TypedDict):
    variable: str
    key: str
    description: str


class Triple(TypedDict):
    subject: NodeId
    relation: NodeId
    object: NodeId


class InputTripleCandidate(BaseModel):
    symbol_mappings: list[SymbolMapping]
    triple: Triple


class OutputTripleCandidate(InputTripleCandidate):
    filter_score: float


class InputNodeCandidate(BaseModel):
    node_id: NodeId


class OutputNodeCandidate(InputNodeCandidate):
    filter_score: float
