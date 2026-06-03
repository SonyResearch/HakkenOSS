from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from hakken_ml_toolkit.ml_utils import DSVUtils


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    # Create a temporary CSV file for testing
    csv_content = """name,age,score
Alice,30,95.5
Bob,25,87.3
Charlie,35,92.1
"""
    csv_file = tmp_path / "sample.csv"
    csv_file.write_text(csv_content)
    return csv_file


def test_read_csv_default(sample_csv: Path) -> None:
    # Test reading CSV without specifying dtype
    df = DSVUtils.read_dsv(sample_csv, header=0)

    expected_df = pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie"],
            "age": [30, 25, 35],
            "score": [95.5, 87.3, 92.1],
        }
    )

    assert_frame_equal(df, expected_df)


def test_read_csv_with_dtype(sample_csv: Path) -> None:
    # Test reading CSV with specified dtype
    dtype: defaultdict[str, Any] = defaultdict(list, {"name": str, "age": int, "score": float})
    df = DSVUtils.read_dsv(sample_csv, header=0, dtype=dtype)

    expected_df = pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie"],
            "age": [30, 25, 35],
            "score": [95.5, 87.3, 92.1],
        }
    ).astype(dtype)

    assert_frame_equal(df, expected_df)


def test_read_csv_file_not_found() -> None:
    # Test behavior when file is not found
    with pytest.raises(FileNotFoundError):
        DSVUtils.read_dsv(Path("non_existent_file.csv"))


def test_write_csv(sample_csv: Path, tmp_path: Path) -> None:
    df = DSVUtils.read_dsv(sample_csv, header=0)
    output_path = tmp_path / "output.csv"
    DSVUtils.write_dsv(df, output_path, delimiter=",")

    df_reread = DSVUtils.read_dsv(output_path, header=0)
    pd.testing.assert_frame_equal(df, df_reread)


def test_stratified_sampling() -> None:
    np.random.seed(42)
    sample_fraction = 0.1
    data = {
        "id": range(1000),
        "category": np.random.choice(["A", "B", "C"], size=1000),
        "value": np.random.randn(1000),
    }
    df = pd.DataFrame(data)

    length_list = []
    for _i in range(100):
        sampled = DSVUtils.stratified_sampling(
            df, stratify_column="category", sample_fraction=sample_fraction
        )
        length_list.append(len(sampled))

    expected_size = int(len(df) * sample_fraction)

    assert np.mean(length_list) == pytest.approx(expected_size, rel=0.1)
