import numpy as np
import polars as pl
from pydantic import BaseModel, Field


class DataBundle(BaseModel):
    """Bundle of preprocessed data for model training.

    Contains tensorized facts data and mapping DataFrames for entities,
    relations, domains, and timestamps. All data is pre-computed and ready
    for training.
    """

    facts_dict: dict[str, np.ndarray] = Field(
        ...,
        description=(
            "Dictionary mapping split names (e.g., 'train', 'val', 'test') "
            "to PyTorch tensors containing fact data"
        ),
    )

    relations_map_df: pl.DataFrame = Field(
        ...,
        description=(
            "Mapping DataFrame with columns 'id' (relation_type) and 'index' (relation_index)"
        ),
    )

    nodes_map_df: pl.DataFrame = Field(
        ...,
        description=(
            "Mapping DataFrame with columns 'id' (node_id), 'index' (node_index), "
            "and 'domain_index'"
        ),
    )

    domains_map_df: pl.DataFrame = Field(
        ...,
        description="Mapping DataFrame with columns 'id' (domain_id) and 'index' (domain_index)",
    )

    timestamps_map_df: pl.DataFrame = Field(
        ...,
        description=(
            "Mapping DataFrame with columns 'id' (timestamp/year) and'index' (timestamp_index)"
        ),
    )

    model_config = {
        "arbitrary_types_allowed": True,  # Allow Polars DataFrames and PyTorch tensors
        "validate_assignment": True,  # Validate on assignment
    }

    def get_num_relations(self) -> int:
        """Get the number of unique relations."""
        return self.relations_map_df.height

    def get_num_nodes(self) -> int:
        """Get the number of unique nodes."""
        return self.nodes_map_df.height

    def get_num_domains(self) -> int:
        """Get the number of unique domains."""
        return self.domains_map_df.height

    def get_num_timestamps(self) -> int:
        """Get the number of unique timestamps."""
        return self.timestamps_map_df.height

    def get_tensor_shape(self, split: str = "train") -> tuple:
        """Get the shape of the tensor for a given split."""
        if split not in self.facts_dict:
            raise ValueError(f"Split '{split}' not found in facts_dict")
        return self.facts_dict[split].shape

    def summary(self) -> dict:
        """Get a summary of the data bundle."""
        return {
            "num_relations": self.get_num_relations(),
            "num_nodes": self.get_num_nodes(),
            "num_domains": self.get_num_domains(),
            "num_timestamps": self.get_num_timestamps(),
            "tensor_shapes": {split: tensor.shape for split, tensor in self.facts_dict.items()},
            "has_timestamps": self.timestamps_map_df is not None,
        }
