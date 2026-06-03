from pydantic import BaseModel

from query_common.entities.conditions.base import ConditionID
from query_common.entities.kg.identifier import ConceptIdentifier
from query_common.entities.variable import Variable, VarLabel


class RequestVariable(BaseModel):
    label: str
    domain: str

    def to_query_variable(self) -> Variable:
        return Variable(label=self.label, domain_identifier=self.domain)


class Candidate(BaseModel):
    var_assignments: dict[VarLabel, ConceptIdentifier]
    condition_scores: dict[ConditionID, float]
    query_score: float | None = None


class QueryRequest(BaseModel):
    formula: str
    variables: list[RequestVariable]
    n_candidates: int


class QueryResponse(BaseModel):
    candidates: list[Candidate]
