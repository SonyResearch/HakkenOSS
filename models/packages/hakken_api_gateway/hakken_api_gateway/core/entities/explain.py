from pydantic import BaseModel, Field


class ExplanationConfig(BaseModel):
    batch_size: int = 32
    type: str = "sufficient"


class ExplanationItem(BaseModel):
    data: str
    length: int
    score: float


# Requests from upstream (UI, program, ...) and to downstream (our services)
class ExplainReqFromUpstream(BaseModel):
    triple: list[str]
    estimated_time: int = Field(alias="estimatedTime")


class ExplainReqToDownstream(BaseModel):
    triples_to_probe: list[list[str]]
    num_explanations: int = 10
    explanation_configs: list[ExplanationConfig]


# Response to upstream (UI, program, ...) and from downstream (our services)
class ExplainResp(BaseModel):
    # The key is a dynamic string (the relationship ID),
    # and the value is a list of explanation items.
    explanations: dict[str, list[ExplanationItem]]


class ExplainLengthReq(BaseModel):
    triples_to_probe: list[list[str]]


class ExplainLengthRes(BaseModel):
    length_dict: dict[str, int]
