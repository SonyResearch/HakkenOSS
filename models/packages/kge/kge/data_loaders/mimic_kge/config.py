import os

from pydantic import BaseModel, ConfigDict, Field, field_validator


def default_scaler_path() -> str | None:
    scaler_folder: str | None = os.getenv("CACHED_DATA_FOLDER", None)

    if scaler_folder is None:
        return None
    return os.path.join(scaler_folder, "target_scaler.json")


class MimicKGEDataLoaderConfig(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        frozen=True,
        extra="forbid",
    )

    # Required fields that must be provided
    num_relations: int = Field(ge=1, description="Number of relations in the knowledge graph")
    num_neighbors: list[int] = Field(description="Number of neighbors to sample per layer")

    # Optional fields with defaults
    batch_size: int = Field(default=1, ge=1, description="Batch size for sampling")
    num_batches_for_scaling: int = Field(
        default=10, ge=1, description="Number of batches for scaler fitting"
    )
    negs_per_pos: int = Field(
        default=1, ge=0, description="Number of negative samples per positive sample"
    )
    corrupt_probs: tuple[float, float, float] = Field(
        default=(1 / 3, 1 / 3, 1 / 3),
        description="Probabilities for corrupting (subject, relation, object)",
    )
    shuffle: bool = Field(default=True, description="Whether to shuffle samples")

    scaler_path: str | None = Field(
        default_factory=default_scaler_path,
        description="File path to save or load the target scaler JSON file.",
    )
    pin_memory: bool = Field(
        default=True,
        description="Whether to pin memory for faster data transfer to GPU in PyTorch DataLoader.",
    )
    num_workers: int = Field(
        default=16,
        ge=0,
        description="Number of subprocesses to use for data loading in PyTorch DataLoader.",
    )

    @field_validator("corrupt_probs")
    @classmethod
    def validate_corrupt_probs(cls, v):
        if len(v) != 3:
            msg = "corrupt_probs must have exactly 3 elements"
            raise ValueError(msg)
        if not all(0 <= p <= 1 for p in v):
            msg = "All probabilities must be between 0 and 1"
            raise ValueError(msg)
        if not abs(sum(v) - 1.0) < 1e-6:
            msg = "Probabilities must sum to 1.0"
            raise ValueError(msg)
        return v

    @field_validator("num_neighbors")
    @classmethod
    def validate_num_neighbors(cls, v):
        if not v:
            msg = "num_neighbors must be a non-empty list"
            raise ValueError(msg)

        return v
