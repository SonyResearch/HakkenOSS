from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import polars as pl
import pytest

from hakken_models.steps.dataset.filter_and_split import (
    filter_and_split_step,
)

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------
# Fixtures with typing
# ---------------------------


@pytest.fixture
def sample_facts_df() -> pl.DataFrame:
    """Return a small, typed facts DataFrame."""
    return pl.DataFrame(
        {
            "subject_id": [1, 2, 3, 4],
            "object_id": [10, 20, 30, 40],
            "relation_type": ["A", "B", "A", "B"],
            "year": [2000, 2005, 2010, 2015],
        }
    )


@pytest.fixture
def sample_nodes_df() -> pl.DataFrame:
    """Return a small nodes DataFrame."""
    return pl.DataFrame(
        {
            "node_id": [1, 2, 3, 4, 10, 20, 30, 40, 999],
            "node_name": ["a", "b", "c", "d", "x", "y", "z", "w", "unused"],
        }
    )


# ---------------------------
# Main tests
# ---------------------------


def test_filter_and_split_basic(
    sample_facts_df: pl.DataFrame,
    sample_nodes_df: pl.DataFrame,
    tmp_path: Path,
) -> None:
    """Test relation filtering, temporal slicing, and node filtering."""

    assert tmp_path is not None

    temporal_partitions = {
        "train": ("2000-01-01", "2010-01-01"),
        "val": ("2010-01-01", "2015-01-01"),
        "test": ("2015-01-01", None),
    }

    allowed_relations = ["A", "B"]

    with patch("polars.DataFrame.write_csv", autospec=True) as _mock_write:
        (
            train_df,
            val_df,
            test_df,
            filtered_nodes_df,
        ) = filter_and_split_step.entrypoint(
            facts_df=sample_facts_df,
            nodes_df=sample_nodes_df,
            allowed_relations=allowed_relations,
            temporal_partitions=temporal_partitions,
        )

    # Type-check the outputs explicitly
    assert isinstance(train_df, pl.DataFrame)
    assert isinstance(val_df, pl.DataFrame)
    assert isinstance(test_df, pl.DataFrame)
    assert isinstance(filtered_nodes_df, pl.DataFrame)

    # -----------------------------
    # 1. Check temporal splits
    # -----------------------------
    assert train_df.height == 2
    assert val_df.height == 1
    assert test_df.height == 1

    assert train_df["year"].to_list() == [2000, 2005]
    assert val_df["year"].to_list() == [2010]
    assert test_df["year"].to_list() == [2015]

    # -----------------------------
    # 2. Node filtering
    # -----------------------------
    used_nodes: list[int] = (
        train_df["subject_id"].to_list()
        + train_df["object_id"].to_list()
        + val_df["subject_id"].to_list()
        + val_df["object_id"].to_list()
        + test_df["subject_id"].to_list()
        + test_df["object_id"].to_list()
    )

    assert set(filtered_nodes_df["node_id"].to_list()) == set(used_nodes)
    assert 999 not in filtered_nodes_df["node_id"].to_list()


def test_missing_val_and_test(
    sample_facts_df: pl.DataFrame,
    sample_nodes_df: pl.DataFrame,
) -> None:
    """If val/test are missing, they should be empty DataFrames."""

    temporal_partitions: dict[str, tuple[str | None, str | None]] = {
        "train": (None, None),
    }

    (
        train_df,
        val_df,
        test_df,
        filtered_nodes_df,
    ) = filter_and_split_step.entrypoint(
        facts_df=sample_facts_df,
        nodes_df=sample_nodes_df,
        allowed_relations=["A", "B"],
        temporal_partitions=temporal_partitions,
    )

    # Types
    assert isinstance(train_df, pl.DataFrame)
    assert isinstance(val_df, pl.DataFrame)
    assert isinstance(test_df, pl.DataFrame)
    assert isinstance(filtered_nodes_df, pl.DataFrame)

    # Expected sizes
    assert train_df.height == sample_facts_df.height
    assert val_df.height == 0
    assert test_df.height == 0

    expected_nodes: list[int] = (
        sample_facts_df["subject_id"].to_list() + sample_facts_df["object_id"].to_list()
    )

    assert set(filtered_nodes_df["node_id"].to_list()) == set(expected_nodes)
