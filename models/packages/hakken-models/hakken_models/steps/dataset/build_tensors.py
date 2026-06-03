from typing import Annotated

import numpy as np
import polars as pl
from loguru import logger
from zenml import ArtifactConfig, log_metadata, step


def create_facts_array(
    facts_df: pl.DataFrame,
    relations_map_df: pl.DataFrame,
    nodes_map_df: pl.DataFrame,
    timestamps_map_df: pl.DataFrame | None = None,
) -> np.ndarray:
    """Converts facts_df into the indexed tensor representation.

    If timestamps_map_df is provided, the output facts_tensor has shape [num_facts, 4].
    Otherwise, facts_tensor has shape [num_facts, 3].

    Args:
        facts_df: Polars DataFrame containing facts with columns for subject, relation, object,
            and optionally timestamp.
        relations_map_df: Mapping DataFrame with columns 'id' and 'index'.
        nodes_map_df: Mapping DataFrame with columns 'id' and 'index'.
        timestamps_map_df: Optional mapping DataFrame with columns 'id' and 'index'.

    Returns:
        PyTorch tensor of shape [num_facts, 3] or [num_facts, 4] with dtype torch.long.
    """
    try:
        # Join with nodes_map_df to get subject indices
        facts_with_indices = facts_df.join(
            nodes_map_df.select(["id", "index"]),
            left_on="subject_id",
            right_on="id",
            how="left",
        ).rename({"index": "subject_idx"})

        # Join with nodes_map_df to get object indices
        facts_with_indices = facts_with_indices.join(
            nodes_map_df.select(["id", "index"]),
            left_on="object_id",
            right_on="id",
            how="left",
        ).rename({"index": "object_idx"})

        # Join with relations_map_df to get relation indices
        facts_with_indices = facts_with_indices.join(
            relations_map_df.select(["id", "index"]),
            left_on="relation_type",
            right_on="id",
            how="left",
        ).rename({"index": "relation_idx"})

        if timestamps_map_df is not None:
            # Join with timestamps_map_df to get timestamp indices
            facts_with_indices = facts_with_indices.join(
                timestamps_map_df.select(["id", "index"]),
                left_on="year",
                right_on="id",
                how="left",
            ).rename({"index": "timestamp_idx"})

            # Select indexed columns and convert to tensor
            indexed_cols = facts_with_indices.select(
                ["subject_idx", "relation_idx", "object_idx", "timestamp_idx"]
            )
            facts_array = indexed_cols.to_numpy().astype(np.int64)
        else:
            # Select indexed columns without timestamp and convert to tensor
            indexed_cols = facts_with_indices.select(["subject_idx", "relation_idx", "object_idx"])
            facts_array = indexed_cols.to_numpy().astype(np.int64)

    except Exception as e:
        logger.error(f"❌ Failed to create facts tensor: {e}")
        raise
    else:
        return facts_array


@step
def build_tensor_step(
    facts_df: pl.DataFrame,
    relations_map_df: pl.DataFrame,
    nodes_map_df: pl.DataFrame,
    timestamps_map_df: pl.DataFrame | None = None,
) -> Annotated[np.ndarray, ArtifactConfig(name="{split_name}_index_np")]:
    """Build a numpy array for a single split (train/val/test)."""

    logger.info(f"🔨 Building array with {facts_df.height} facts...")

    array = create_facts_array(
        facts_df=facts_df,
        relations_map_df=relations_map_df,
        nodes_map_df=nodes_map_df,
        timestamps_map_df=timestamps_map_df,
    )

    # metadata
    expected_shape = (facts_df.height, 4) if timestamps_map_df is not None else (facts_df.height, 3)

    log_metadata(
        metadata={
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "num_facts": facts_df.height,
            "expected_shape": list(expected_shape),
        },
    )

    logger.info(f"✅ Finished building array: shape={array.shape}")
    return array
