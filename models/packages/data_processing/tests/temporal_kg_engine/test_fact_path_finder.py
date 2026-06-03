# mypy: ignore-errors

from typing import Any

import networkx as nx
import pytest

from data_processing.temporal_kg_engine.in_memory import FactPathFinder


class TestFactPathFinder:
    """Comprehensive test suite for FactPathFinder functionality."""

    @pytest.fixture
    def simple_graph(self) -> nx.MultiDiGraph:
        """Create a simple graph: A --[knows]--> B --[likes]--> C"""
        graph = nx.MultiDiGraph()
        graph.add_edge("A", "B", key="knows")
        graph.add_edge("B", "C", key="likes")
        return graph

    @pytest.fixture
    def bidirectional_graph(self) -> nx.MultiDiGraph:
        """Create a graph with bidirectional edges."""
        graph = nx.MultiDiGraph()
        graph.add_edge("A", "B", key="knows")
        graph.add_edge("B", "A", key="knows")
        graph.add_edge("B", "C", key="likes")
        graph.add_edge("C", "B", key="likes")
        return graph

    @pytest.fixture
    def multi_relation_graph(self) -> nx.MultiDiGraph:
        """Create a graph with multiple relation types between nodes."""
        graph = nx.MultiDiGraph()
        graph.add_edge("A", "B", key="knows")
        graph.add_edge("A", "B", key="likes")
        graph.add_edge("A", "B", key="trusts")
        graph.add_edge("B", "C", key="knows")
        graph.add_edge("B", "C", key="hates")
        return graph

    @pytest.fixture
    def complex_graph(self) -> nx.MultiDiGraph:
        """Create a more complex graph with multiple paths."""
        graph = nx.MultiDiGraph()
        # Path 1: A -> B -> C
        graph.add_edge("A", "B", key="knows")
        graph.add_edge("B", "C", key="likes")
        # Path 2: A -> D -> C
        graph.add_edge("A", "D", key="knows")
        graph.add_edge("D", "C", key="knows")
        # Bidirectional edges
        graph.add_edge("B", "A", key="knows")
        graph.add_edge("C", "B", key="dislikes")
        return graph

    @pytest.fixture
    def disconnected_graph(self) -> nx.MultiDiGraph:
        """Create a graph with disconnected components."""
        graph = nx.MultiDiGraph()
        graph.add_edge("A", "B", key="knows")
        graph.add_edge("C", "D", key="likes")
        return graph

    # ===== Test Initialization =====

    def test_init_valid_graph(self, simple_graph: nx.MultiDiGraph) -> None:
        """Test initialization with a valid MultiDiGraph."""
        finder = FactPathFinder(simple_graph)
        assert finder.graph is simple_graph
        assert finder.relation_type_attr == "relation_type"

    def test_init_custom_relation_attr(self, simple_graph: nx.MultiDiGraph) -> None:
        """Test initialization with custom relation_type_attr."""
        finder = FactPathFinder(simple_graph, relation_type_attr="custom_attr")
        assert finder.relation_type_attr == "custom_attr"

    # ===== Test get_known_relations =====

    def test_get_known_relations_existing_edge(self, simple_graph: nx.MultiDiGraph) -> None:
        """Test getting relations for an existing edge."""
        finder = FactPathFinder(simple_graph)
        relations = finder.get_known_relations("A", "B")
        assert relations == {"knows"}

    def test_get_known_relations_non_existing_edge(self, simple_graph: nx.MultiDiGraph) -> None:
        """Test getting relations for a non-existing edge."""
        finder = FactPathFinder(simple_graph)
        relations = finder.get_known_relations("A", "C")
        assert relations == set()

    def test_get_known_relations_multiple_relations(
        self, multi_relation_graph: nx.MultiDiGraph
    ) -> None:
        """Test getting multiple relations between same nodes."""
        finder = FactPathFinder(multi_relation_graph)
        relations = finder.get_known_relations("A", "B")
        assert relations == {"knows", "likes", "trusts"}

    def test_get_known_relations_reverse_direction(
        self, bidirectional_graph: nx.MultiDiGraph
    ) -> None:
        """Test that reverse direction is checked separately."""
        finder = FactPathFinder(bidirectional_graph)
        relations_forward = finder.get_known_relations("A", "B")
        relations_reverse = finder.get_known_relations("B", "A")
        assert relations_forward == {"knows"}
        assert relations_reverse == {"knows"}

    # ===== Test has_edge =====

    def test_has_edge_existing(self, simple_graph: nx.MultiDiGraph) -> None:
        """Test has_edge for an existing edge."""
        finder = FactPathFinder(simple_graph)
        assert finder.has_edge("A", "B", "knows") is True

    def test_has_edge_non_existing(self, simple_graph: nx.MultiDiGraph) -> None:
        """Test has_edge for a non-existing edge."""
        finder = FactPathFinder(simple_graph)
        assert finder.has_edge("A", "B", "likes") is False
        assert finder.has_edge("A", "C", "knows") is False

    def test_has_edge_wrong_direction(self, simple_graph: nx.MultiDiGraph) -> None:
        """Test has_edge checks direction correctly."""
        finder = FactPathFinder(simple_graph)
        assert finder.has_edge("A", "B", "knows") is True
        assert finder.has_edge("B", "A", "knows") is False

    # ===== Test node_path_to_facts - Basic Cases =====

    def test_node_path_to_facts_empty_path(self, simple_graph: nx.MultiDiGraph) -> None:
        """Test with empty node path."""
        finder = FactPathFinder(simple_graph)
        result = finder.node_path_to_facts([])
        assert result == []

    def test_node_path_to_facts_single_node(self, simple_graph: nx.MultiDiGraph) -> None:
        """Test with single node path."""
        finder = FactPathFinder(simple_graph)
        result = finder.node_path_to_facts(["A"])
        assert result == []

    def test_node_path_to_facts_simple_path(self, simple_graph: nx.MultiDiGraph) -> None:
        """Test simple two-node path."""
        finder = FactPathFinder(simple_graph)
        result = finder.node_path_to_facts(["A", "B"], include_reverse_each_hop=False)
        assert len(result) == 1
        assert result[0] == [("A", "knows", "B")]

    def test_node_path_to_facts_three_node_path(self, simple_graph: nx.MultiDiGraph) -> None:
        """Test three-node path."""
        finder = FactPathFinder(simple_graph)
        result = finder.node_path_to_facts(["A", "B", "C"], include_reverse_each_hop=False)
        assert len(result) == 1
        assert result[0] == [("A", "knows", "B"), ("B", "likes", "C")]

    # ===== Test node_path_to_facts - Reverse Edges =====

    def test_node_path_to_facts_with_reverse(self, bidirectional_graph: nx.MultiDiGraph) -> None:
        """Test including reverse edges at each hop."""
        finder = FactPathFinder(bidirectional_graph)
        result = finder.node_path_to_facts(["A", "B"], include_reverse_each_hop=True)

        # Should have paths for both A->B and B->A
        assert len(result) == 2
        # Check that both directions are possible
        fact_tuples = [tuple(path[0]) for path in result]
        assert ("A", "knows", "B") in fact_tuples and ("B", "knows", "A") in fact_tuples

    def test_node_path_to_facts_without_reverse(self, bidirectional_graph: nx.MultiDiGraph) -> None:
        """Test excluding reverse edges."""
        finder = FactPathFinder(bidirectional_graph)
        result = finder.node_path_to_facts(["A", "B"], include_reverse_each_hop=False)

        assert len(result) == 1
        assert result[0] == [("A", "knows", "B")]

    # ===== Test node_path_to_facts - Multiple Relations =====

    def test_node_path_to_facts_multiple_relations(
        self, multi_relation_graph: nx.MultiDiGraph
    ) -> None:
        """Test path with multiple relation types."""
        finder = FactPathFinder(multi_relation_graph)
        result = finder.node_path_to_facts(["A", "B", "C"], include_reverse_each_hop=False)

        # Should have 3 relations A->B * 2 relations B->C = 6 combinations
        assert len(result) == 6

        # Verify all combinations exist
        relation_pairs = [(path[0][1], path[1][1]) for path in result]
        expected_pairs = [
            ("knows", "knows"),
            ("knows", "hates"),
            ("likes", "knows"),
            ("likes", "hates"),
            ("trusts", "knows"),
            ("trusts", "hates"),
        ]
        assert sorted(relation_pairs) == sorted(expected_pairs)

    # ===== Test node_path_to_facts - Allowed Relations =====

    def test_node_path_to_facts_allowed_relations_filter(
        self, multi_relation_graph: nx.MultiDiGraph
    ) -> None:
        """Test filtering with allowed_relations."""
        finder = FactPathFinder(multi_relation_graph)
        result = finder.node_path_to_facts(
            ["A", "B", "C"], allowed_relations=["knows"], include_reverse_each_hop=False
        )

        assert len(result) == 1
        assert result[0] == [("A", "knows", "B"), ("B", "knows", "C")]

    def test_node_path_to_facts_allowed_relations_empty_result(
        self, multi_relation_graph: nx.MultiDiGraph
    ) -> None:
        """Test when allowed_relations filters out all paths."""
        finder = FactPathFinder(multi_relation_graph)
        result = finder.node_path_to_facts(
            ["A", "B", "C"], allowed_relations=["nonexistent"], include_reverse_each_hop=False
        )

        assert result == []

    def test_node_path_to_facts_allowed_relations_partial(
        self, multi_relation_graph: nx.MultiDiGraph
    ) -> None:
        """Test partial filtering with allowed_relations."""
        finder = FactPathFinder(multi_relation_graph)
        result = finder.node_path_to_facts(
            ["A", "B", "C"], allowed_relations=["knows", "likes"], include_reverse_each_hop=False
        )

        # Should have: (knows, knows), (knows, hates), (likes, knows), (likes, hates)
        # But hates is not allowed, so: (knows, knows), (likes, knows)
        assert len(result) == 2

    # ===== Test node_path_to_facts - Max Paths =====

    def test_node_path_to_facts_max_paths_limit(
        self, multi_relation_graph: nx.MultiDiGraph
    ) -> None:
        """Test max_paths parameter limits results."""
        finder = FactPathFinder(multi_relation_graph)
        result = finder.node_path_to_facts(
            ["A", "B", "C"], max_paths=3, include_reverse_each_hop=False
        )

        assert len(result) == 3

    def test_node_path_to_facts_max_paths_zero(self, multi_relation_graph: nx.MultiDiGraph) -> None:
        """Test max_paths=0 returns empty list immediately."""
        finder = FactPathFinder(multi_relation_graph)
        result = finder.node_path_to_facts(
            ["A", "B", "C"], max_paths=0, include_reverse_each_hop=False
        )

        assert result == []

    def test_node_path_to_facts_max_paths_larger_than_total(
        self, simple_graph: nx.MultiDiGraph
    ) -> None:
        """Test max_paths larger than total paths."""
        finder = FactPathFinder(simple_graph)
        result = finder.node_path_to_facts(
            ["A", "B"], max_paths=100, include_reverse_each_hop=False
        )

        assert len(result) == 1

    # ===== Test node_path_to_facts - Error Cases =====

    def test_node_path_to_facts_disconnected_nodes(
        self, disconnected_graph: nx.MultiDiGraph
    ) -> None:
        """Test path between disconnected nodes raises RuntimeError."""
        finder = FactPathFinder(disconnected_graph)

        with pytest.raises(RuntimeError, match="Edge data for"):
            finder.node_path_to_facts(["A", "C"])

    def test_node_path_to_facts_nonexistent_nodes(self) -> None:
        """Test path with nonexistent nodes raises RuntimeError."""
        graph = nx.MultiDiGraph()
        graph.add_edge("A", "B", key="knows")
        finder = FactPathFinder(graph)

        with pytest.raises(RuntimeError, match="Edge data for"):
            finder.node_path_to_facts(["A", "Z"])

    # ===== Test node_path_to_facts - Complex Scenarios =====

    def test_node_path_to_facts_long_path(self) -> None:
        """Test longer path with multiple hops."""
        graph = nx.MultiDiGraph()
        graph.add_edge("A", "B", key="r1")
        graph.add_edge("B", "C", key="r2")
        graph.add_edge("C", "D", key="r3")
        graph.add_edge("D", "E", key="r4")

        finder = FactPathFinder(graph)
        result = finder.node_path_to_facts(
            ["A", "B", "C", "D", "E"], include_reverse_each_hop=False
        )

        assert len(result) == 1
        assert result[0] == [("A", "r1", "B"), ("B", "r2", "C"), ("C", "r3", "D"), ("D", "r4", "E")]

    def test_node_path_to_facts_combinatorial_explosion(self) -> None:
        """Test handling of large combinatorial spaces."""
        graph = nx.MultiDiGraph()
        # Create 3 relations for each hop
        for i in range(3):
            graph.add_edge("A", "B", key=f"r{i}")
            graph.add_edge("B", "C", key=f"r{i}")
            graph.add_edge("C", "D", key=f"r{i}")

        finder = FactPathFinder(graph)
        result = finder.node_path_to_facts(["A", "B", "C", "D"], include_reverse_each_hop=False)

        # Should be 3^3 = 27 combinations
        assert len(result) == 27

    def test_node_path_to_facts_fact_direction_combinations(
        self, bidirectional_graph: nx.MultiDiGraph
    ) -> None:
        """Test that all fact direction combinations are generated correctly."""
        finder = FactPathFinder(bidirectional_graph)
        result = finder.node_path_to_facts(["A", "B"], include_reverse_each_hop=True)

        # With bidirectional edges, should have multiple combinations
        assert len(result) >= 1

        # Verify facts are properly formatted as (source, relation, target)
        for path in result:
            for fact in path:
                assert len(fact) == 3
                assert isinstance(fact[0], str)  # source
                assert isinstance(fact[1], str)  # relation
                assert isinstance(fact[2], str)  # target

    # ===== Integration Tests =====

    def test_full_workflow_simple(self, simple_graph: nx.MultiDiGraph) -> None:
        """Test complete workflow with a simple graph."""
        finder = FactPathFinder(simple_graph)

        # Check initial state
        assert finder.has_edge("A", "B", "knows")
        assert finder.get_known_relations("A", "B") == {"knows"}

        # Generate fact paths
        result = finder.node_path_to_facts(["A", "B", "C"], include_reverse_each_hop=False)

        assert len(result) == 1
        assert len(result[0]) == 2  # Two hops
        assert result[0][0] == ("A", "knows", "B")
        assert result[0][1] == ("B", "likes", "C")

    def test_full_workflow_complex(self, complex_graph: nx.MultiDiGraph) -> None:
        """Test complete workflow with a complex graph."""
        finder = FactPathFinder(complex_graph)

        # Test multiple paths exist
        assert finder.has_edge("A", "B", "knows")
        assert finder.has_edge("A", "D", "knows")

        # Test path through B
        result_b = finder.node_path_to_facts(["A", "B", "C"], include_reverse_each_hop=False)
        assert len(result_b) >= 1

        # Test path through D
        result_d = finder.node_path_to_facts(["A", "D", "C"], include_reverse_each_hop=False)
        assert len(result_d) >= 1


# ===== Parametrized Tests =====


class TestFactPathFinderParametrized:
    """Parametrized tests for comprehensive coverage."""

    @pytest.fixture
    def bidirectional_graph(self) -> nx.MultiDiGraph:
        """Create a graph with bidirectional edges."""
        graph = nx.MultiDiGraph()
        graph.add_edge("A", "B", key="knows")
        graph.add_edge("B", "A", key="knows")
        graph.add_edge("B", "C", key="likes")
        graph.add_edge("C", "B", key="likes")
        return graph

    @pytest.mark.parametrize(
        "node_count,expected",
        [
            (0, []),
            (1, []),
            (2, 1),  # At least 1 path
            (3, 1),  # At least 1 path
        ],
    )
    def test_path_length_variations(self, node_count: int, expected: int | list[Any]) -> None:
        """Test various path lengths."""
        graph = nx.MultiDiGraph()
        nodes = [chr(65 + i) for i in range(node_count)]  # A, B, C, ...

        for i in range(len(nodes) - 1):
            graph.add_edge(nodes[i], nodes[i + 1], key="rel")

        finder = FactPathFinder(graph)
        result = finder.node_path_to_facts(nodes, include_reverse_each_hop=False)

        if isinstance(expected, list):
            assert result == expected
        else:
            assert len(result) >= expected

    @pytest.mark.parametrize("include_reverse", [True, False])
    def test_reverse_parameter(
        self, bidirectional_graph: nx.MultiDiGraph, include_reverse: bool
    ) -> None:
        """Test include_reverse_each_hop parameter."""
        finder = FactPathFinder(bidirectional_graph)
        result = finder.node_path_to_facts(["A", "B"], include_reverse_each_hop=include_reverse)

        assert len(result) >= 1
        if not include_reverse:
            # Should only have forward edge
            assert len(result) == 1
            assert result[0][0][0] == "A"  # Source is A


# ===== Edge Case Tests =====


class TestFactPathFinderEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_self_loop(self) -> None:
        """Test graph with self-loop."""
        graph = nx.MultiDiGraph()
        graph.add_edge("A", "A", key="self")

        finder = FactPathFinder(graph)
        result = finder.node_path_to_facts(["A", "A"], include_reverse_each_hop=False)

        assert len(result) == 1
        assert result[0] == [("A", "self", "A")]

    def test_parallel_edges_same_relation(self) -> None:
        """Test handling of parallel edges with same relation key."""
        graph = nx.MultiDiGraph()
        graph.add_edge("A", "B", key="knows")
        # MultiDiGraph allows same key, but it overwrites
        graph.add_edge("A", "B", key="knows")

        finder = FactPathFinder(graph)
        relations = finder.get_known_relations("A", "B")

        # Should still be one unique relation
        assert relations == {"knows"}

    def test_unicode_node_names(self) -> None:
        """Test with unicode node names."""
        graph = nx.MultiDiGraph()
        graph.add_edge("αλφα", "βήτα", key="γνωρίζω")

        finder = FactPathFinder(graph)
        result = finder.node_path_to_facts(["αλφα", "βήτα"], include_reverse_each_hop=False)

        assert len(result) == 1
        assert result[0] == [("αλφα", "γνωρίζω", "βήτα")]

    def test_numeric_string_nodes(self) -> None:
        """Test with numeric string node IDs."""
        graph = nx.MultiDiGraph()
        graph.add_edge("1", "2", key="follows")
        graph.add_edge("2", "3", key="follows")

        finder = FactPathFinder(graph)
        result = finder.node_path_to_facts(["1", "2", "3"], include_reverse_each_hop=False)

        assert len(result) == 1
        assert result[0] == [("1", "follows", "2"), ("2", "follows", "3")]
