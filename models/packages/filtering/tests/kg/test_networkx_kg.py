import networkx as nx
import pytest
from pydantic import ValidationError

from filtering.core.entities.config.knowledge_graph import NetworkXKnowledgeGraphConfig
from filtering.core.entities.kg import EdgeDirection, YearRange
from filtering.impl.kg.networkx_kg import NetworkXKnowledgeGraph


class TestNetworkXKnowledgeGraphConfig:
    def test_init_error_cases(self, nodes_path, edges_path):
        with pytest.raises(ValidationError):
            NetworkXKnowledgeGraphConfig()
        with pytest.raises(ValidationError):
            NetworkXKnowledgeGraphConfig(nodes_path=nodes_path)
        with pytest.raises(ValidationError):
            NetworkXKnowledgeGraphConfig(edges_path=edges_path)


class TestNetworkXKnowledgeGraph:
    def test_from_nodes_and_edges_path(self, nodes_path, edges_path):
        config = NetworkXKnowledgeGraphConfig(nodes_path=nodes_path, edges_path=edges_path)
        graph = NetworkXKnowledgeGraph(config)

        assert isinstance(graph._g, nx.MultiDiGraph)

        node_ids = ["201000002884", "208000021018", "190000021581"]

        degrees = graph.get_degrees(node_ids, direction=EdgeDirection.IN)
        assert degrees == [0, 1, 3]
        degrees = graph.get_degrees(node_ids, direction=EdgeDirection.OUT)
        assert degrees == [2, 1, 45]
        degrees = graph.get_degrees(node_ids, direction=EdgeDirection.ALL)
        assert degrees == [2, 2, 48]

        degrees = graph.get_degrees(
            node_ids, direction=EdgeDirection.IN, year_range=YearRange(2015, 2020)
        )
        assert degrees == [0, 0, 2]
        degrees = graph.get_degrees(
            node_ids, direction=EdgeDirection.OUT, year_range=YearRange(2015, 2020)
        )
        assert degrees == [0, 1, 13]
        degrees = graph.get_degrees(
            node_ids, direction=EdgeDirection.ALL, year_range=YearRange(2015, 2020)
        )
        assert degrees == [0, 1, 15]

        with pytest.raises(KeyError):
            graph.get_degrees([*node_ids, "_UNKNOWN_NODE_"], direction=EdgeDirection.IN)

    def test_from_nodes_and_edges_path_timestamp(self, nodes_path, edges_path):
        config = NetworkXKnowledgeGraphConfig(
            nodes_path=nodes_path,
            edges_path=edges_path,
        )
        graph = NetworkXKnowledgeGraph(config)

        assert isinstance(graph._g, nx.MultiDiGraph)

        node_ids = ["201000002884", "208000021018", "190000021581"]

        degrees = graph.get_degrees(node_ids, direction=EdgeDirection.IN)
        assert degrees == [0, 1, 3]
        degrees = graph.get_degrees(node_ids, direction=EdgeDirection.OUT)
        assert degrees == [2, 1, 45]
        degrees = graph.get_degrees(node_ids, direction=EdgeDirection.ALL)
        assert degrees == [2, 2, 48]

        degrees = graph.get_degrees(
            node_ids, direction=EdgeDirection.IN, year_range=YearRange(2015, 2020)
        )
        assert degrees == [0, 0, 2]
        degrees = graph.get_degrees(
            node_ids, direction=EdgeDirection.OUT, year_range=YearRange(2015, 2020)
        )
        assert degrees == [0, 1, 13]
        degrees = graph.get_degrees(
            node_ids, direction=EdgeDirection.ALL, year_range=YearRange(2015, 2020)
        )
        assert degrees == [0, 1, 15]

        with pytest.raises(KeyError):
            graph.get_degrees([*node_ids, "_UNKNOWN_NODE_"], direction=EdgeDirection.IN)

    def test_from_nodes_and_edges_path_num_occurrences(self, nodes_path, edges_path):
        config = NetworkXKnowledgeGraphConfig(
            nodes_path=nodes_path,
            edges_path=edges_path,
        )
        graph = NetworkXKnowledgeGraph(config)

        assert isinstance(graph._g, nx.MultiDiGraph)

        node_ids = ["201000002884", "208000021018", "190000021581"]

        degrees = graph.get_degrees(node_ids, direction=EdgeDirection.IN)
        assert degrees == [0, 1, 3]
        degrees = graph.get_degrees(node_ids, direction=EdgeDirection.OUT)
        assert degrees == [2, 1, 45]
        degrees = graph.get_degrees(node_ids, direction=EdgeDirection.ALL)
        assert degrees == [2, 2, 48]

        degrees = graph.get_degrees(
            node_ids, direction=EdgeDirection.IN, year_range=YearRange(2015, 2020)
        )
        assert degrees == [0, 0, 2]
        degrees = graph.get_degrees(
            node_ids, direction=EdgeDirection.OUT, year_range=YearRange(2015, 2020)
        )
        assert degrees == [0, 1, 13]
        degrees = graph.get_degrees(
            node_ids, direction=EdgeDirection.ALL, year_range=YearRange(2015, 2020)
        )
        assert degrees == [0, 1, 15]

        with pytest.raises(KeyError):
            graph.get_degrees([*node_ids, "_UNKNOWN_NODE_"], direction=EdgeDirection.IN)
