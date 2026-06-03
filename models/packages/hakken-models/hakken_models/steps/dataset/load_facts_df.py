from typing import Annotated

import polars as pl
from loguru import logger
from zenml import ArtifactConfig, log_metadata, step


@step
def load_facts_df(
    facts_df_path: str,
    separator: str = "\t",
    schema_overrides: dict[str, pl.DataType] | None = None,
) -> Annotated[pl.DataFrame, ArtifactConfig(name="facts_df")]:
    """Load the temporal KG edges (facts) into a Polars DataFrame.

    Args:
        facts_df_path: Path to the facts/edges CSV (S3 or local).
        separator: Column separator used in the edges file.
        schema_overrides: Optional schema override dict for Polars.

    Returns:
        A Polars DataFrame containing edges/facts.
    """

    # Default schema for temporal edges
    default_schema = {
        "subject_id": pl.Utf8,
        "subject_domain": pl.Utf8,
        "relation_type": pl.Utf8,
        "object_id": pl.Utf8,
        "object_domain": pl.Utf8,
        "year": pl.Int16,
        "number_of_occurrences": pl.Int32,
    }

    schema = schema_overrides or default_schema

    # Handle S3 or local paths transparently
    path = facts_df_path
    if not facts_df_path.startswith("s3://"):
        path = f"s3://{facts_df_path}"

    try:
        logger.info(f"📥 Loading facts (edges) from: {path}")

        df = pl.read_csv(
            path,
            separator=separator,
            schema_overrides=schema,
        )

        # Log metadata into ZenML
        log_metadata(
            artifact_name="facts_df",
            metadata={
                "path": path,
                "num_rows": df.height,
                "num_columns": df.width,
                "schema": {col: str(df[col].dtype) for col in df.columns},
            },
            infer_artifact=True,
        )

        logger.info(f"Loaded facts_df with {df.height} rows and {df.width} columns.")

    except Exception as e:
        logger.error(f"❌ Failed to load facts from {path}: {e}")
        raise
    else:
        return df
