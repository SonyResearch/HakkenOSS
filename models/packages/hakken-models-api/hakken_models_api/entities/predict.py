from pydantic import BaseModel, Field


class ModelPredictRequest(BaseModel):
    subject_id_list: list[str] = Field(..., description="List of subject entity IDs")
    object_id_list: list[str] = Field(..., description="List of object entity IDs")
    relation_id_list: list[str] | None = Field(
        None, description="Optional relation IDs to filter predictions"
    )
    inference_config: dict | None = Field(default=None, description="Optional inference parameters")


class ModelPredictResponse(BaseModel):
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
