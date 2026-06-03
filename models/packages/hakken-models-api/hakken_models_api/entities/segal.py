"""SeGAL-specific API entities."""

from hakken_models.models.segal.text_schemas import EntityText, RelationText
from pydantic import BaseModel, Field


class SeGALInfoResponse(BaseModel):
    """Dataset and model metadata for a loaded SeGAL model."""

    num_entities: int = Field(description="Number of entities in the dataset")
    num_relations: int = Field(description="Number of relations in the dataset")
    has_embeddings: bool = Field(
        description="Whether pre-computed node and relation embeddings are available"
    )
    embedding_dim: int | None = Field(
        default=None,
        description="Dimensionality of embeddings, or None if not available",
    )


class FactTextItem(BaseModel):
    """A single fact as text: (subject, relation, object, optional timestamp)."""

    subject: EntityText
    relation: RelationText
    object: EntityText
    timestamp: float | None = None


class ScoreTextRequest(BaseModel):
    """Request body for text-based fact scoring."""

    target_facts: list[FactTextItem] = Field(description="Facts to score.")
    context_facts: list[FactTextItem] | None = Field(
        default=None,
        description="Optional context facts used to build the KG for the GNN.",
    )
    inference_config: dict | None = Field(
        default=None,
        description="Optional inference parameters (encode_batch_size, return_probs).",
    )


class ScoreTextResponse(BaseModel):
    """Response for text-based fact scoring."""

    scores_list: list[float] = Field(description="Logit scores for each target fact.")
    normalized_scores_list: list[float] | None = Field(
        default=None,
        description="Sigmoid-normalized probabilities, or None if return_probs=False.",
    )
