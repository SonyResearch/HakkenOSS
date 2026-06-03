from pydantic import BaseModel


class ExplanationAPI(BaseModel):
    data: str | None = None
    length: int | None = None
    score: float | None = None


Hypothesis = str


class PathExplainerResponse(BaseModel):
    explanations: dict[Hypothesis, list[ExplanationAPI]]


class PathExplainerLengthResponse(BaseModel):
    length_dict: dict[Hypothesis, int]
