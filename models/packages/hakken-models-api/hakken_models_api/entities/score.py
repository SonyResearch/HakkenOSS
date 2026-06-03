from pydantic import BaseModel, Field


class ModelScoreRequest(BaseModel):
    facts_list: list[tuple[str, str, str]] = Field(
        description="List of facts represented as tuple of IDs"
    )
    inference_config: dict | None = Field(default=None, description="Optional inference parameters")


class ModelScoreResponse(BaseModel):
    scores_list: list[float] = Field(description="Scoring results for each input fact")
    normalized_scores_list: list[float] | None = Field(
        description="Normalized scores for each input fact. None if normalize=False in the request"
    )
