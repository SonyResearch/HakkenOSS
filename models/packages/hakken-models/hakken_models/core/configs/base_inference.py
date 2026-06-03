"""Base inference config shared by temporal KG models."""

from pydantic import BaseModel, Field, field_validator

from hakken_models.core.constants import MissingPolicy


class BaseInferenceConfig(BaseModel):
    """Shared inference config for temporal KG link prediction models."""

    split_names: list[str] = Field(
        default_factory=lambda: ["train", "val", "test"],
        description="Dataset splits used to build the temporal KG context.",
    )
    return_probs: bool = Field(
        default=True,
        description="Whether to return probabilities in addition to logits.",
    )
    num_neighbors: list[int] = Field(
        default_factory=lambda: [512, 512],
        description="Number of neighbors to sample per GNN layer.",
    )
    on_missing: MissingPolicy = Field(
        default=MissingPolicy.ZERO,
        description=(
            "How to handle missing nodes/relations: "
            "'raise' throws an error, 'zero' assigns zero score/probability."
        ),
    )

    @field_validator("num_neighbors")
    @classmethod
    def check_positive_neighbors(cls, v: list[int]) -> list[int]:
        if any(n <= 0 for n in v):
            raise ValueError("All num_neighbors must be positive integers.")
        return v
