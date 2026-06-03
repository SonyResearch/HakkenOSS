from typing import Annotated

import polars as pl
from loguru import logger
from zenml import ArtifactConfig, log_metadata, step


@step
def build_domain_mapping(
    nodes_df: pl.DataFrame,
) -> Annotated[pl.DataFrame, ArtifactConfig(name="domains_mapping_df")]:
    """Build mapping DataFrame for domains.

    The unique domains are extracted from nodes_df node_domain_id.

    Args:
        nodes_df: Polars DataFrame containing nodes with 'node_domain_id' column.

    Returns:
        Polars DataFrame with columns 'id' (str | int) and 'index' (int):
        Mapping of unique domain IDs to indices.
    """
    try:
        logger.info("🔨 Building domain mapping DataFrame...")

        # Extract unique domain IDs and assign indices
        domain_mapping_df = (
            nodes_df.select("node_domain_id")
            .unique()
            .with_row_index(name="index")
            .rename({"node_domain_id": "id"})
        )

        log_metadata(
            artifact_name="domains_mapping_df",
            metadata={
                "num_rows": domain_mapping_df.height,
                "num_columns": domain_mapping_df.width,
                "schema": {
                    col: str(domain_mapping_df[col].dtype) for col in domain_mapping_df.columns
                },
            },
            infer_artifact=True,
        )
        logger.info(f"✅ Generated domain mapping with {domain_mapping_df.height} unique domains.")

    except Exception as e:
        logger.error(f"❌ Failed to build domain mapping DataFrame: {e}")
        raise
    else:
        return domain_mapping_df


@step
def build_nodes_mapping(
    nodes_df: pl.DataFrame,
    domains_mapping_df: pl.DataFrame,
) -> Annotated[pl.DataFrame, ArtifactConfig(name="nodes_mapping_df")]:
    """Build mapping DataFrame for nodes with domain indices.

    The unique entities are extracted from nodes_df node_id.
    Each node is also associated with its domain_index from the domain mapping.

    Args:
        nodes_df: Polars DataFrame containing nodes with 'node_id' and 'node_domain_id'
            columns.
        domain_mapping_df: Polars DataFrame with 'id' (domain_id) and 'index' (domain_index)
            columns.

    Returns:
        Polars DataFrame with columns:
        - 'id' (str | int): node_id
        - 'index' (int): node index
        - 'domain_index' (int): domain index for this node
    """
    try:
        logger.info("🔨 Building nodes mapping DataFrame with domain indices...")

        # Extract unique node IDs with their domain IDs
        unique_nodes = nodes_df.select(["node_id", "node_domain_id"]).unique()

        # Join with domain mapping to get domain_index
        nodes_with_domain = unique_nodes.join(
            domains_mapping_df.select(["id", "index"]).rename(
                {"id": "node_domain_id", "index": "domain_index"}
            ),
            on="node_domain_id",
            how="left",
        )

        # Add node index and rename columns
        nodes_mapping_df = (
            nodes_with_domain.with_row_index(name="index")
            .select(["node_id", "index", "domain_index"])
            .rename({"node_id": "id"})
        )

        log_metadata(
            artifact_name="nodes_mapping_df",
            metadata={
                "num_rows": nodes_mapping_df.height,
                "num_columns": nodes_mapping_df.width,
                "schema": {
                    col: str(nodes_mapping_df[col].dtype) for col in nodes_mapping_df.columns
                },
            },
            infer_artifact=True,
        )
        logger.info(f"✅ Generated nodes mapping with {nodes_mapping_df.height} indices.")

    except Exception as e:
        logger.error(f"❌ Failed to build nodes mapping DataFrame: {e}")
        raise
    else:
        return nodes_mapping_df


@step
def build_relation_mapping(
    facts_df: pl.DataFrame,
) -> Annotated[pl.DataFrame, ArtifactConfig(name="relations_mapping_df")]:
    """Build mapping DataFrame for relations.

    The unique relations are extracted from facts_df relation_type (str).

    Args:
        facts_df: Polars DataFrame containing facts with 'relation_type' column.

    Returns:
        Polars DataFrame with columns 'id' (str) and 'index' (int):
        Mapping of unique relation types to indices.
    """
    try:
        logger.info("🔨 Building relation mapping DataFrame...")

        # Extract unique relation types and assign indices
        relation_mapping_df = (
            facts_df.select("relation_type")
            .unique()
            .with_row_index(name="index")
            .rename({"relation_type": "id"})
        )

        log_metadata(
            artifact_name="relations_mapping_df",
            metadata={
                "num_rows": relation_mapping_df.height,
                "num_columns": relation_mapping_df.width,
                "schema": {
                    col: str(relation_mapping_df[col].dtype) for col in relation_mapping_df.columns
                },
            },
            infer_artifact=True,
        )
        logger.info(
            f"✅ Generated relation mapping with {relation_mapping_df.height} unique relations."
        )

    except Exception as e:
        logger.error(f"❌ Failed to build relation mapping DataFrame: {e}")
        raise
    else:
        return relation_mapping_df


@step
def build_timestamp_mapping(
    facts_df: pl.DataFrame,
) -> Annotated[pl.DataFrame, ArtifactConfig(name="timestamps_mapping_df")]:
    """Build mapping DataFrame for timestamps.

    The unique timestamps are extracted from facts_df year (int).

    Args:
        facts_df: Polars DataFrame containing facts with 'year' column.

    Returns:
        Polars DataFrame with columns 'id' (int) and 'index' (int):
        Mapping of unique timestamps to indices.
    """
    try:
        logger.info("🔨 Building timestamp mapping DataFrame...")

        # Extract unique timestamps and assign indices (sorted for deterministic ordering)
        timestamp_mapping_df = (
            facts_df.select("year")
            .unique()
            .sort("year")
            .with_row_index(name="index")
            .rename({"year": "id"})
        )

        log_metadata(
            artifact_name="timestamps_mapping_df",
            metadata={
                "num_rows": timestamp_mapping_df.height,
                "num_columns": timestamp_mapping_df.width,
                "schema": {
                    col: str(timestamp_mapping_df[col].dtype)
                    for col in timestamp_mapping_df.columns
                },
            },
            infer_artifact=True,
        )
        logger.info(
            f"✅ Generated timestamp mapping with {timestamp_mapping_df.height} unique timestamps."
        )

    except Exception as e:
        logger.error(f"❌ Failed to build timestamp mapping DataFrame: {e}")
        raise
    else:
        return timestamp_mapping_df
