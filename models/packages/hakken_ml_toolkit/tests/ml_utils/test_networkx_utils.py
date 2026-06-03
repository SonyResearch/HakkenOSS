import pytest

try:
    import networkx as nx
except ImportError:
    pytest.skip("NetworkX is not installed", allow_module_level=True)


import pandas as pd
import pytest
from networkx.algorithms.isomorphism import is_isomorphic

from hakken_ml_toolkit.ml_utils.networkx import NetworkXUtils, NetworkXUtilsConfig


@pytest.fixture
def sample_df():
    # Sample DataFrame
    data = {
        "ocid_subject": ["2", "1", "1", "1", "3", "5", "4"],
        "ocid_object": ["1", "2", "3", "4", "4", "4", "2"],
        "relation_type": [
            "AFFECTS",
            "TREATS",
            "TREATS",
            "INDUCES",
            "IS",
            "AFFECTS",
            "IS",
        ],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_multi_di_graph(sample_df):
    # Sample DataFrame
    config = NetworkXUtilsConfig(
        source_column="ocid_subject",
        target_column="ocid_object",
        relation_column="relation_type",
        multiple_edges=True,
        directed=True,
    )

    return NetworkXUtils.load_graph_from_pandas(sample_df, config)


@pytest.fixture
def sample_multi_graph(sample_df):
    # Sample DataFrame
    config = NetworkXUtilsConfig(
        source_column="ocid_subject",
        target_column="ocid_object",
        relation_column="relation_type",
        multiple_edges=True,
        directed=False,
    )

    return NetworkXUtils.load_graph_from_pandas(sample_df, config)


def test_conversion_to_undirected(sample_multi_graph, sample_multi_di_graph):
    result = NetworkXUtils.convert_graph_to_undirected(sample_multi_di_graph, multiple_edges=True)

    # Check is equivalent to creating a uniderected graph from a dataframe
    assert is_isomorphic(result, sample_multi_graph)  # same structure
    assert set(result.nodes) == set(sample_multi_graph.nodes)
    assert set(result.edges) == set(sample_multi_graph.edges)

    # Check that is different from the directed graph, but the nodes are preserved
    with pytest.raises(nx.exception.NetworkXError):
        is_isomorphic(result, sample_multi_di_graph)  # different type

    assert set(result.nodes) == set(sample_multi_di_graph.nodes)
    assert set(result.edges) != set(sample_multi_di_graph.edges)


def test_shortest_path_length(sample_multi_graph, sample_multi_di_graph):
    graph, graph_u = sample_multi_di_graph, sample_multi_graph
    source = "2"
    target = "4"

    # Directed
    length = NetworkXUtils.get_shortest_path_length(graph, source, target, include_extrema=True)
    assert length == 4

    # Undirected
    length = NetworkXUtils.get_shortest_path_length(graph_u, source, target, include_extrema=True)
    assert length == 3

    # Counting the links and not the nodes
    length = NetworkXUtils.get_shortest_path_length(graph, source, target, include_extrema=False)
    assert length == 2
    length = NetworkXUtils.get_shortest_path_length(graph_u, source, target, include_extrema=False)
    assert length == 1

    # No path available
    source = "4"
    target = "5"
    length = NetworkXUtils.get_shortest_path_length(graph, source, target, include_extrema=True)
    assert length == -1


def test_shortest_path_finder(sample_multi_di_graph, sample_multi_graph):
    # Directed
    shortest_path_list = NetworkXUtils.all_shortest_paths(
        sample_multi_di_graph, source="3", target="2"
    )
    expected_list = [[("3", {"relation": "IS"}, "4"), ("4", {"relation": "IS"}, "2")]]

    assert shortest_path_list == expected_list

    # Undirected
    shortest_path_list = NetworkXUtils.all_shortest_paths(
        sample_multi_graph, source="3", target="2"
    )
    expected_list = [
        [("3", {"relation": "TREATS"}, "1"), ("1", {"relation": "AFFECTS"}, "2")],
        [("3", {"relation": "TREATS"}, "1"), ("1", {"relation": "TREATS"}, "2")],
        [("3", {"relation": "IS"}, "4"), ("4", {"relation": "IS"}, "2")],
    ]
    assert shortest_path_list == expected_list

    # No path found
    shortest_path_list = NetworkXUtils.all_shortest_paths(
        sample_multi_di_graph, source="4", target="5"
    )
    assert shortest_path_list == []
