from typing import Annotated

import polars as pl
from zenml import ArtifactConfig, step


@step
def extract_metadata_step(
    nodes_map_df: pl.DataFrame,
    relations_map_df: pl.DataFrame,
    domains_map_df: pl.DataFrame,
    timestamps_map_df: pl.DataFrame,
) -> Annotated[dict, ArtifactConfig(name="dataset_metadata")]:
    """Extract metadata needed for model creation."""

    metadata = {
        "num_entities": nodes_map_df.height,
        "num_relations": relations_map_df.height,
    }

    metadata["num_domains"] = domains_map_df.height

    metadata["num_timestamps"] = timestamps_map_df.height

    return metadata
