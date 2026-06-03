from typing import Annotated

import polars as pl
from loguru import logger
from zenml import ArtifactConfig, step
from zenml.integrations.polars.materializers import PolarsMaterializer


# ----------------------------
# Utility
# ----------------------------
def temporal_slice(
    frame: pl.DataFrame,
    start: str | None,
    end: str | None,
) -> pl.DataFrame:
    sliced = frame
    if start:
        sliced = sliced.filter(pl.col("date") >= pl.lit(start).str.strptime(pl.Date))
    if end:
        sliced = sliced.filter(pl.col("date") < pl.lit(end).str.strptime(pl.Date))
    return sliced


# ============================================================
#  STEP 1: Filter + split (ARTIFACTS ONLY — no manual S3 paths)
# ============================================================
@step(
    output_materializers={
        "train_facts_df": PolarsMaterializer,
        "val_facts_df": PolarsMaterializer,
        "test_facts_df": PolarsMaterializer,
        "filtered_nodes_df": PolarsMaterializer,
    },
)
def filter_and_split_step(
    facts_df: pl.DataFrame,
    nodes_df: pl.DataFrame,
    allowed_relations: list[str] | None = None,
    temporal_partitions: dict[str, tuple[str | None, str | None]] | None = None,
) -> tuple[
    Annotated[pl.DataFrame, ArtifactConfig(name="train_facts_df")],
    Annotated[pl.DataFrame, ArtifactConfig(name="val_facts_df")],
    Annotated[pl.DataFrame, ArtifactConfig(name="test_facts_df")],
    Annotated[pl.DataFrame, ArtifactConfig(name="filtered_nodes_df")],
]:
    """Filter facts and nodes, then split facts into temporal partitions.

    NOTE:
        This step returns artifacts only.
        It does NOT manually save to S3. Export happens in a separate step.
    """

    logger.info("🚀 Starting filter_and_split_step")

    # ------------------------------------------------------
    # defaults
    # ------------------------------------------------------
    if temporal_partitions is None:
        temporal_partitions = {}
        temporal_partitions["train"] = (None, None)

    if "train" not in temporal_partitions:
        raise ValueError(
            f"'train' temporal partition is required. "
            f"Provided partitions: {list(temporal_partitions.keys())}"
        )

    df = facts_df

    # ------------------------------------------------------
    # 1. Filter by allowed relations
    # ------------------------------------------------------
    if allowed_relations:
        logger.info(f"Filtering facts by allowed relations: {allowed_relations}")
        df = df.filter(pl.col("relation_type").is_in(allowed_relations))
        logger.info(f"Remaining facts after relation filtering: {df.height}")

    # ------------------------------------------------------
    # 2. Convert year to date for temporal slicing
    # ------------------------------------------------------
    if "year" not in df.columns:
        raise ValueError("facts_df must contain a 'year' column for temporal splits.")

    df = (
        df.with_columns(pl.col("year").cast(pl.Int32))
        .with_columns((pl.col("year").cast(str) + "-01-01").alias("date_str"))
        .with_columns(pl.col("date_str").str.strptime(pl.Date, strict=False).alias("date"))
        .drop("date_str")
    )

    # ------------------------------------------------------
    # 3. Temporal splits
    # ------------------------------------------------------
    logger.info("📅 Applying temporal splits...")

    train_bounds = temporal_partitions["train"]
    val_bounds = temporal_partitions.get("val", None)
    test_bounds = temporal_partitions.get("test", None)

    train_df = temporal_slice(df, train_bounds[0], train_bounds[1])

    val_df = temporal_slice(df, val_bounds[0], val_bounds[1]) if val_bounds else df.head(0)
    test_df = temporal_slice(df, test_bounds[0], test_bounds[1]) if test_bounds else df.head(0)

    logger.info(
        f"Split sizes → train: {train_df.height}, val: {val_df.height}, test: {test_df.height}"
    )

    # ------------------------------------------------------
    # 4. Filter nodes based on used ids
    # ------------------------------------------------------
    logger.info("🔍 Filtering nodes_df to include only nodes present in splits.")

    used_ids = (
        pl.concat(
            [
                train_df["subject_id"],
                train_df["object_id"],
                val_df["subject_id"],
                val_df["object_id"],
                test_df["subject_id"],
                test_df["object_id"],
            ]
        )
        .unique()
        .to_list()
    )

    filtered_nodes_df = nodes_df.filter(pl.col("node_id").is_in(used_ids))

    logger.info(f"Filtered nodes: {filtered_nodes_df.height} out of {nodes_df.height} originally.")

    logger.info("✔ Completed filter_and_split_step")

    # ZenML will persist these automatically to S3 artifact store
    return train_df, val_df, test_df, filtered_nodes_df
