from pydantic import BaseModel, Field


class KGEResponse(BaseModel):
    success: bool
    message: str | None = None


# ===================================================
# ===================================================


class KGEPredictRequest(BaseModel):
    subject_id_list: list[str] = Field(..., description="List of subject entity IDs")
    object_id_list: list[str] = Field(..., description="List of object entity IDs")
    relation_id_list: list[str] | None = Field(
        None, description="Optional relation IDs to filter predictions"
    )
    inference_config: dict | None = Field(default=None, description="Optional inference parameters")


class KGEPredictResponse(BaseModel):
    relations_ids: list[str] = Field(
        description=(
            "The relation IDs for which scores were computed."
            "This matches the input relation_id_list from the request."
        )
    )
    relations_probs: list[list[float]] | None = Field(
        None,
        description=(
            "Normalized probability scores in a nested list structure. "
            "Outer list length equals the number of subject-object pairs. "
            "Inner list length equals the number of relation IDs. "
            "relations_probs[i][j] gives the probability score for the j-th relation "
            "and the i-th (subject, object) pair. Only present if the KGE model has a scaler."
        ),
    )
    relations_scores: list[list[float]] | None = Field(
        None,
        description=(
            "Raw model scores in a nested list structure. "
            "Outer list length equals the number of subject-object pairs. "
            "Inner list length equals the number of relation IDs. "
            "relations_scores[i][j] gives the raw score for the j-th relation "
            "and the i-th (subject, object) pair."
        ),
    )


# ===================================================
# ===================================================


class KGEScoreIndexRequest(BaseModel):
    facts_index_list: list[list[int]] = Field(
        description="List of triples represented as integer lists"
    )
    normalize: bool = Field(
        default=False,
        description="Whether to normalize the output scores between [0.0, 1.0]",
    )


class KGEScoreRequest(BaseModel):
    facts_list: list[tuple[str, str, str]] = Field(
        description="List of facts represented as tuple of IDs"
    )
    normalize: bool = Field(
        default=False,
        description="Whether to normalize the output scores between [0.0, 1.0]",
    )


class KGEScoreResponse(BaseModel):
    scores_list: list[float] = Field(description="Scoring results for each input fact")
    normalized_scores_list: list[float] | None = Field(
        description="Normalized scores for each input fact. None if normalize=False in the request"
    )


# ===================================================
# ===================================================


class KGESFitScoreScalerRequest(BaseModel):
    loader_kwargs: dict | None = None
    overwrite: bool = False
