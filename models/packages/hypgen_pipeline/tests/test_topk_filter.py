import pandas as pd
import pytest

from hypgen_pipeline.impl.topk_filter import TopKFilter, TopKFilterConfig


@pytest.fixture
def sample_df():
    # Sample DataFrame
    data = {
        "node_pair": [
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


def test_split_node_pairs(sample_df):
    result = TopKFilter._split_node_pairs(sample_df)
    expected = pd.DataFrame(
        {
            "node1": ["A", "A", "A", "B", "B", "B", "C", "C"],
            "node2": ["S", "G", "G", "A", "D", "L", "M", "N"],
            "node_pair": [
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
    )
    assert expected.sort_index(axis=1).equals(result.sort_index(axis=1))


def test_topk_filter(sample_df):
    result = TopKFilter.filter(sample_df, TopKFilterConfig(topk=2))
    expected = pd.DataFrame(
        {
            "node_pair": [
                ["A", "G"],
                ["A", "G"],
                ["B", "D"],
                ["B", "L"],
                ["C", "M"],
                ["C", "N"],
                ["A", "S"],
            ],
            "confidence_score": [0.95, 0.90, 0.85, 0.80, 0.95, 0.70, 0.75],
            "other_column": ["Z", "Y", "V", "U", "T", "S", "X"],
        }
    )

    assert expected.sort_index(axis=1).equals(result.sort_index(axis=1))
