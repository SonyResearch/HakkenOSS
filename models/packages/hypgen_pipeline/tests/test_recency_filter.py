import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from hypgen_pipeline.impl.recency_filter import RecencyFilter, RecencyFilterConfig


@pytest.fixture
def sample_hypothesis():
    # Sample DataFrame
    data = {
        "node_pair_ocids": [
            ["A", "S"],
            ["A", "G"],
            ["A", "G"],
            ["B", "A"],
            ["B", "D"],
            ["B", "L"],
            ["C", "M"],
            ["C", "N"],
        ],
        "confidence_score": [0.75, 0.90, 0.95, 0.60, 0.85, 0.80, 0.95, 0.70],
        "other_column": ["X", "Y", "Z", "W", "V", "U", "T", "S"],
    }

    return pd.DataFrame(data)


@pytest.fixture
def sample_recency_dict():
    # Sample DataFrame
    return {
        "A": {"mode": 2022, "median": 2018},
        "S": {"mode": 2022, "median": 2019},
        "G": {"mode": 2022, "median": 2002},
        "B": {"mode": 2022, "median": 2002},
        "D": {"mode": 2022, "median": 2002},
        "L": {"mode": 2022, "median": 2022},
        "C": {"mode": 2022, "median": 2022},
        "M": {"mode": 2022, "median": 2022},
        "N": {"mode": 2022, "median": 2010},
    }


@pytest.fixture
def sample_papers_count_dict():
    # Sample DataFrame
    return {
        "A": 1,
        "S": 2,
        "G": 3,
        "B": 4,
        "D": 5,
        "L": 6,
        "C": 7,
        "M": 8,
        "N": 9,
    }


def test_recency_filter(sample_recency_dict, sample_papers_count_dict, sample_hypothesis):
    config = RecencyFilterConfig(
        median_year=2019,
        entities_research_year_statistics=sample_recency_dict,
        entities_papers_count=sample_papers_count_dict,
    )
    df_filtered = RecencyFilter.filter(sample_hypothesis, config)
    expected = pd.DataFrame(
        {
            "node_pair_ocids": [["A", "S"], ["B", "L"], ["C", "M"], ["C", "N"]],
            "confidence_score": [0.75, 0.80, 0.95, 0.70],
            "other_column": ["X", "U", "T", "S"],
            "recency_mode": [[2022, 2022], [2022, 2022], [2022, 2022], [2022, 2022]],
            "recency_median": [[2018, 2019], [2002, 2022], [2022, 2022], [2022, 2010]],
            "papers_count": [[1, 2], [4, 6], [7, 8], [7, 9]],
        }
    )

    assert_frame_equal(df_filtered, expected)
