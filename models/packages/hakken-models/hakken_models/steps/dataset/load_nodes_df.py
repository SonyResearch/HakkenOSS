from typing import Annotated

import polars as pl
from loguru import logger
from zenml import ArtifactConfig, log_metadata, step


@step
def load_nodes_df(
    nodes_df_path: str, separator: str = "\t", schema_overrides: dict | None = None
) -> Annotated[pl.DataFrame, ArtifactConfig(name="nodes_df")]:
    """Load a nodes CSV file (S3 or local) into a Polars DataFrame.

    Args:
        nodes_df_path: S3 or local path to nodes CSV (without prefix).
        separator: Column separator.
        schema_overrides: Optional Polars schema overrides.

    Returns:
        Polars DataFrame containing the nodes.
    """

    if schema_overrides is None:
        schema_overrides = {
            "node_id": pl.Utf8,
            "node_domain": pl.Utf8,
            "node_name": pl.Utf8,
            "node_domain_id": pl.Utf8,
        }

    path = nodes_df_path
    if not nodes_df_path.startswith("s3://"):
        path = f"s3://{nodes_df_path}"

    try:
        logger.info(f"📥 Loading nodes from: {path}")

        df = pl.read_csv(
            path,
            separator=separator,
            schema_overrides={
                "node_id": pl.Utf8,
                "node_domain": pl.Utf8,
                "node_name": pl.Utf8,
                "node_domain_id": pl.Utf8,
            },
        )
        log_metadata(
            artifact_name="nodes_df",
            metadata={
                "path": path,
                "num_rows": df.height,
                "num_columns": df.width,
                "schema": {col: str(df[col].dtype) for col in df.columns},
            },
            infer_artifact=True,
        )
        logger.info(f"Loaded DataFrame with {df.height} rows, {df.width} columns.")

    except Exception as e:
        logger.error(f"❌ Failed to load nodes from {path}: {e}")
        raise
    else:
        return df
