import pandas as pd
import pytest
from hakken_ml_toolkit.ml_utils.networkx import NetworkXUtils, NetworkXUtilsConfig
from pandas.testing import assert_frame_equal

from hypgen_pipeline.impl.path_length_filter import PathLengthFilter, PathLengthFilterConfig


@pytest.fixture
def sample_graph():
    # Sample DataFrame
    data = {
        "ocid_subject": ["2", "1", "1", "1", "3", "5", "4"],
        "ocid_object": ["1", "2", "3", "4", "4", "4", "2"],
        "relation_type": ["AFFECTS", "TREATS", "TREATS", "INDUCES", "IS", "AFFECTS", "IS"],
    }

    nxutils_config = NetworkXUtilsConfig(
        source_column="ocid_subject",
        target_column="ocid_object",
        relation_column="relation_type",
        multiple_edges=True,
        directed=False,
    )

    return NetworkXUtils.load_graph_from_pandas(pd.DataFrame(data), config=nxutils_config)


def test_shortest_path_filter(sample_graph):
    hypothesis = pd.DataFrame({"node_pairs": [["1", "5"], ["2", "3"], ["2", "1"], ["3", "4"]]})

    # Default
    config = PathLengthFilterConfig(
        node_pair_ocids_column="node_pairs",
        reference_kg=sample_graph,
        min_path_length=None,  # default
        max_path_length=None,  # default
        include_extrema=False,  # default
    )

    filtered_df = PathLengthFilter.filter(hypothesis, config)

    expected = pd.DataFrame(
        {
            "node_pairs": [["1", "5"], ["2", "3"], ["2", "1"], ["3", "4"]],
            "shortest_path_length": [2, 2, 1, 1],
        }
    )
    print(filtered_df)

    assert_frame_equal(filtered_df, expected)

    # Filtered without extrema
    config = PathLengthFilterConfig(
        node_pair_ocids_column="node_pairs",
        reference_kg=sample_graph,
        min_path_length=2,
        max_path_length=None,  # default
        include_extrema=False,  # default
    )

    expected = pd.DataFrame(
        {
            "node_pairs": [
                ["1", "5"],
                ["2", "3"],
            ],
            "shortest_path_length": [2, 2],
        }
    )

    filtered_df = PathLengthFilter.filter(hypothesis, config)

    assert_frame_equal(filtered_df, expected)

    # Filtered with extrema
    config = PathLengthFilterConfig(
        node_pair_ocids_column="node_pairs",
        reference_kg=sample_graph,
        min_path_length=2,
        max_path_length=None,  # default
        include_extrema=True,  # default
    )

    filtered_df = PathLengthFilter.filter(hypothesis, config)

    expected = pd.DataFrame(
        {
            "node_pairs": [["1", "5"], ["2", "3"], ["2", "1"], ["3", "4"]],
            "shortest_path_length": [4, 4, 3, 3],
        }
    )

    assert_frame_equal(filtered_df, expected)
