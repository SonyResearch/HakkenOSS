from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class QueryVariable(BaseModel):
    label: str
    domain: str


class QueryApi(BaseModel):
    variables: list[QueryVariable]
    formula: str


class GraphNode(BaseModel):
    is_variable: bool = Field(alias="isVariable")
    label: str
    domain: str
    id: str

    model_config = ConfigDict(populate_by_name=True)


class Condition(BaseModel):
    head: GraphNode
    tail: GraphNode
    relation: str


class Hypothesis(BaseModel):
    # 'condition' and 'addValue' are standard camelCase conversions
    condition: Condition
    add_value: str = Field(alias="addValue")
    condition_type: str = Field(alias="conditionType")

    model_config = ConfigDict(populate_by_name=True)


class QueryId(BaseModel):
    id: str


class UserId(BaseModel):
    id: str


class QueryUserId(BaseModel):
    id: str
    user_id: str


class SearchParameters(BaseModel):
    beam_size: int


class Candidate(BaseModel):
    var_assignments: dict[str, str]
    condition_scores: dict[str, float]
    query_score: float


# DB Model to store queries
class QueryDBModel(BaseModel):
    query: QueryApi
    query_string: str

    # Hypotheses is a Dict because the key "0" implies dynamic keys (ids/indices)
    hypotheses: dict[str, Hypothesis]

    # Constraints is empty in example, assuming Dict[str, Any]
    constraints: dict[str, Any]

    candidates_number: int
    query_mode: str

    user_id: str
    query_id: str

    model_config = ConfigDict(populate_by_name=False)


# Requests from upstream (UI, program, ...) and to downstream (our services)
class QueryReqFromUpstream(BaseModel):
    query_api: QueryApi = Field(alias="queryApi")
    query_string: str = Field(alias="queryString")

    # Hypotheses is a Dict because the key "0" implies dynamic keys (ids/indices)
    hypotheses: dict[str, Hypothesis]

    # Constraints is empty in example, assuming Dict[str, Any]
    constraints: dict[str, Any]

    candidates_number: int = Field(alias="candidatesNumber")
    query_mode: str = Field(alias="queryMode")

    # This config allows you to instantiate the model using
    # either the JSON name (queryApi) or the Python name (query_api)
    model_config = ConfigDict(populate_by_name=True)


class QueryReqToDownstream(BaseModel):
    formula: str
    variables: list[QueryVariable]
    n_candidates: int
    search_algorithm: str = "beam"
    search_parameters: SearchParameters


# Response to upstream (UI, program, ...) and from downstream (our services)
class QueryRespToUpstream(BaseModel):
    queries: list[QueryDBModel]


class QueryRespFromDownstream(BaseModel):
    candidates: list[Candidate]
