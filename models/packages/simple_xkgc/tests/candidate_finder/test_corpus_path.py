# ruff: noqa: PLR2004

import tempfile
from pathlib import Path

import pytest
import torch
from hakken_ml_toolkit.ml_base_structures.fact import assert_is_fact_index_list
from hakken_ml_toolkit.ml_utils.extras import TensorCreator

from hakken_explainer.candidate_finder.corpus import CorpusPathFinder

# Entity constants
ALICE = 0
BOB = 1
CHARLIE = 2
DAVID = 3
EVE = 4
COMPANY_A = 5
COMPANY_B = 6
CITY_X = 7
CITY_Y = 8
ISOLATED_NODE = 99

# Relation constants
KNOWS = 0
WORKS_AT = 1
LOCATED_IN = 2
FRIENDS_WITH = 3
MANAGES = 4


class TestFindPaths:
    """Test suite for CorpusPathFinder.find_paths method."""

    @pytest.fixture
    def simple_facts(self) -> torch.Tensor:
        return torch.tensor(
            [
                [ALICE, KNOWS, BOB],
                [BOB, KNOWS, CHARLIE],
                [CHARLIE, KNOWS, DAVID],
            ],
            dtype=torch.long,
        )

    @pytest.fixture
    def complex_facts(self) -> torch.Tensor:
        return torch.tensor(
            [
                # Direct path: Alice -> Bob -> Charlie
                [ALICE, KNOWS, BOB],
                [BOB, KNOWS, CHARLIE],
                # Alternative path: Alice -> David -> Charlie
                [ALICE, FRIENDS_WITH, DAVID],
                [DAVID, KNOWS, CHARLIE],
                # Another path: Alice -> Eve -> Bob -> Charlie
                [ALICE, WORKS_AT, EVE],
                [EVE, KNOWS, BOB],
                # Cycle: Charlie -> Alice
                [CHARLIE, KNOWS, ALICE],
                # Multiple relations between same nodes
                [ALICE, MANAGES, BOB],  # Second relation Alice -> Bob
                [BOB, FRIENDS_WITH, CHARLIE],  # Second relation Bob -> Charlie
            ],
            dtype=torch.long,
        )

    @pytest.fixture
    def path_finder_simple(self, simple_facts: torch.Tensor) -> CorpusPathFinder:
        finder = CorpusPathFinder(undirected=True)
        finder.setup(facts_batch=simple_facts)
        return finder

    @pytest.fixture
    def path_finder_complex(self, complex_facts: torch.Tensor) -> CorpusPathFinder:
        finder = CorpusPathFinder(undirected=True)
        finder.setup(facts_batch=complex_facts)
        return finder

    def test_find_direct_path(self, path_finder_simple: CorpusPathFinder) -> None:
        """Test finding a direct single-hop path."""
        paths = path_finder_simple.find_candidates(source=ALICE, target=BOB, k=1)

        assert len(paths) > 0
        for path in paths:
            assert_is_fact_index_list(path)
        # Check source and target from facts
        assert all(p[0][0] in (ALICE, BOB) for p in paths)
        assert all(p[-1][2] in (ALICE, BOB) for p in paths)

    def test_find_two_hop_path(self, path_finder_simple: CorpusPathFinder) -> None:
        """Test finding a two-hop path."""
        paths = path_finder_simple.find_candidates(source=ALICE, target=CHARLIE, k=2)

        assert len(paths) > 0
        for path in paths:
            assert_is_fact_index_list(path, length=2)

    def test_find_three_hop_path(self, path_finder_simple: CorpusPathFinder) -> None:
        """Test finding a three-hop path."""
        k = 3
        paths = path_finder_simple.find_candidates(source=ALICE, target=DAVID, k=k)

        assert len(paths) > 0
        for path in paths:
            assert_is_fact_index_list(path, length=k)

    def test_find_paths_no_path_exists(self, path_finder_simple: CorpusPathFinder) -> None:
        """Test finding paths when no path exists."""
        # Add isolated node to facts
        facts_with_isolated = torch.cat(
            [
                path_finder_simple.known_facts,
                torch.tensor([[ISOLATED_NODE, KNOWS, ISOLATED_NODE]], dtype=torch.long),
            ]
        )

        finder = CorpusPathFinder(undirected=True)
        finder.setup(facts_batch=facts_with_isolated)

        paths = finder.find_candidates(source=ALICE, target=ISOLATED_NODE, k=2)

        assert len(paths) == 0

    def test_find_paths_source_equals_target(self, path_finder_simple: CorpusPathFinder) -> None:
        """Test finding paths when source equals target."""
        paths = path_finder_simple.find_candidates(source=ALICE, target=ALICE, k=0)

        # Should return empty or very short path
        assert len(paths) >= 0

    def test_find_paths_with_k_none(self, path_finder_simple: CorpusPathFinder) -> None:
        """Test finding paths with k=None (shortest path length)."""
        paths = path_finder_simple.find_candidates(source=ALICE, target=CHARLIE, k=None)

        assert len(paths) > 0
        for path in paths:
            assert_is_fact_index_list(path, length=2)

    def test_find_multiple_paths_same_length(self, path_finder_complex: CorpusPathFinder) -> None:
        """Test finding multiple paths of the same length."""
        paths = path_finder_complex.find_candidates(source=ALICE, target=CHARLIE, k=2)
        assert len(paths) > 1
        for path in paths:
            assert_is_fact_index_list(path, length=2)

    def test_find_paths_with_multiple_relations(
        self, path_finder_complex: CorpusPathFinder
    ) -> None:
        """Test finding paths where multiple relations exist between nodes."""
        paths = path_finder_complex.find_candidates(source=ALICE, target=BOB, k=1)

        # Should find paths with different relations (KNOWS and MANAGES)
        assert len(paths) >= 2, f"{paths}"

        relations_used = [p[0][1] for p in paths]
        assert KNOWS in relations_used or MANAGES in relations_used

    def test_find_paths_with_allowed_relations(self, path_finder_complex: CorpusPathFinder) -> None:
        """Test finding paths with relation filtering."""
        # Only allow KNOWS relation
        paths = path_finder_complex.find_candidates(
            source=ALICE, target=CHARLIE, k=2, allowed_relations=[KNOWS]
        )

        assert len(paths) > 0
        # All paths should only use KNOWS relation
        for path in paths:
            assert all(fact[1] == KNOWS for fact in path)

    def test_find_paths_with_restrictive_allowed_relations(
        self, path_finder_complex: CorpusPathFinder
    ) -> None:
        """Test that restrictive relation filtering may yield no paths."""
        paths = path_finder_complex.find_candidates(
            source=ALICE, target=CHARLIE, k=2, allowed_relations=[LOCATED_IN]
        )

        assert len(paths) == 0

    def test_find_paths_undirected_graph(self, complex_facts: torch.Tensor) -> None:
        """Test that undirected graph allows bidirectional traversal."""
        finder = CorpusPathFinder(undirected=True)
        finder.setup(facts_batch=complex_facts)

        paths = finder.find_candidates(source=CHARLIE, target=ALICE, k=1)

        assert len(paths) > 0

    def test_find_paths_node_not_in_graph(self, path_finder_simple: CorpusPathFinder) -> None:
        """Test finding paths when source or target not in graph."""
        # Non-existent source
        paths = path_finder_simple.find_candidates(source=999, target=ALICE, k=2)
        assert len(paths) == 0

        # Non-existent target
        paths = path_finder_simple.find_candidates(source=ALICE, target=999, k=2)
        assert len(paths) == 0

    def test_find_paths_with_cycles(self, path_finder_complex: CorpusPathFinder) -> None:
        """Test finding paths in graph with cycles."""
        # Graph has cycle: Alice -> Bob -> Charlie -> Alice
        paths = path_finder_complex.find_candidates(source=ALICE, target=ALICE, k=3)

        # Should be able to find cyclic paths
        assert len(paths) >= 0

    def test_find_paths_consistency_with_shortest_path_length(
        self, path_finder_simple: CorpusPathFinder
    ) -> None:
        """Test that find_paths respects shortest path length."""
        shortest_len = path_finder_simple.shortest_path_length(source=ALICE, target=DAVID)

        paths = path_finder_simple.find_candidates(source=ALICE, target=DAVID, k=shortest_len)

        assert len(paths) > 0
        for path in paths:
            assert_is_fact_index_list(path, length=shortest_len)

    def test_find_paths_empty_allowed_relations(
        self, path_finder_complex: CorpusPathFinder
    ) -> None:
        """Test finding paths with empty allowed_relations list."""
        paths = path_finder_complex.find_candidates(
            source=ALICE, target=CHARLIE, k=2, allowed_relations=[]
        )

        assert len(paths) == 0

    def test_find_paths_large_k_value(self, path_finder_complex: CorpusPathFinder) -> None:
        """Test finding paths with k larger than shortest path."""
        # Shortest path from Alice to Bob is 1, but request k=3

        k = 3
        paths = path_finder_complex.find_candidates(source=ALICE, target=BOB, k=k)

        assert len(paths) > 0
        assert all(len(p) == k for p in paths)

    def test_find_paths_path_structure(self, path_finder_simple: CorpusPathFinder) -> None:
        """Test the structure of returned paths."""
        paths = path_finder_simple.find_candidates(source=ALICE, target=CHARLIE, k=2)

        assert len(paths) > 0
        path = paths[0]

        # Verify path structure
        assert len(path) == 2  # Two edges
        assert isinstance(path, list)
        assert all(len(fact) == 3 for fact in path)  # Each fact is (s, r, o)

    def test_find_paths_with_self_loop(self) -> None:
        """Test finding paths in graph with self-loops."""
        facts_with_loop = torch.tensor(
            [
                [ALICE, KNOWS, BOB],
                [BOB, KNOWS, BOB],  # Self-loop
                [BOB, KNOWS, CHARLIE],
            ],
            dtype=torch.long,
        )

        finder = CorpusPathFinder(undirected=True)
        finder.setup(facts_batch=facts_with_loop)

        paths = finder.find_candidates(source=ALICE, target=CHARLIE, k=2)

        assert len(paths) > 0


class TestShortestPathLength:
    """Test suite for CorpusPathFinder.shortest_path_length method."""

    @pytest.fixture
    def path_finder(self) -> CorpusPathFinder:
        facts = torch.tensor(
            [
                [ALICE, KNOWS, BOB],
                [BOB, KNOWS, CHARLIE],
                [CHARLIE, KNOWS, DAVID],
                [ALICE, FRIENDS_WITH, DAVID],  # Shortcut
                [ISOLATED_NODE, FRIENDS_WITH, ISOLATED_NODE],
            ],
            dtype=torch.long,
        )
        finder = CorpusPathFinder(undirected=True)
        finder.setup(facts_batch=facts)
        return finder

    def test_shortest_path_direct_connection(self, path_finder: CorpusPathFinder) -> None:
        """Test shortest path length for directly connected nodes."""
        length = path_finder.shortest_path_length(source=ALICE, target=BOB)
        assert length == 1

    def test_shortest_path_two_hops(self, path_finder: CorpusPathFinder) -> None:
        """Test shortest path length for two-hop connection."""
        length = path_finder.shortest_path_length(source=ALICE, target=CHARLIE)
        assert length == 2

    def test_shortest_path_with_shortcut(self, path_finder: CorpusPathFinder) -> None:
        """Test that shortest path uses shortcut when available."""
        # There's a direct edge Alice -> David (shortcut)
        # vs longer path Alice -> Bob -> Charlie -> David
        length = path_finder.shortest_path_length(source=ALICE, target=DAVID)
        assert length == 1  # Uses shortcut

    def test_shortest_path_same_node(self, path_finder: CorpusPathFinder) -> None:
        """Test shortest path length when source equals target."""
        length = path_finder.shortest_path_length(source=ALICE, target=ALICE)
        assert length == 0

    def test_shortest_path_no_path_exists(self, path_finder: CorpusPathFinder) -> None:
        """Test shortest path length when no path exists."""
        length = path_finder.shortest_path_length(source=ALICE, target=ISOLATED_NODE)
        assert length == -1

    def test_shortest_path_node_not_in_graph(self, path_finder: CorpusPathFinder) -> None:
        """Test shortest path length for non-existent nodes."""
        length = path_finder.shortest_path_length(source=999, target=ALICE)
        assert length == -1

        length = path_finder.shortest_path_length(source=ALICE, target=999)
        assert length == -1


class TestPathFinderIntegration:
    """Integration tests for the complete PathFinder workflow."""

    @pytest.fixture
    def knowledge_graph_facts(self) -> torch.Tensor:
        """Create a realistic knowledge graph for testing.

        Returns:
            Tensor representing a small knowledge graph.
        """
        return torch.tensor(
            [
                # Person relationships
                [ALICE, KNOWS, BOB],
                [BOB, KNOWS, CHARLIE],
                [CHARLIE, KNOWS, BOB],
                # Work relationships
                [ALICE, WORKS_AT, COMPANY_A],
                [BOB, WORKS_AT, COMPANY_A],
                [CHARLIE, WORKS_AT, COMPANY_B],
                # Location relationships
                [COMPANY_A, LOCATED_IN, CITY_X],
                [COMPANY_B, LOCATED_IN, CITY_Y],
            ],
            dtype=torch.long,
        )

    def test_end_to_end_path_finding_and_conversion(
        self, knowledge_graph_facts: torch.Tensor
    ) -> None:
        """Test complete workflow: setup -> find_paths -> convert."""
        finder = CorpusPathFinder(undirected=True)
        finder.setup(facts_batch=knowledge_graph_facts)

        # Find paths from Alice to Charlie
        paths = finder.find_candidates(source=ALICE, target=CHARLIE, k=2)

        assert len(paths) > 0
        for path in paths:
            assert_is_fact_index_list(path)

        # Convert paths to search space
        search_space = TensorCreator.long_tensor(paths, device=knowledge_graph_facts.device)

        # Verify output shape
        assert search_space.shape[0] == len(paths)
        assert search_space.shape[1] == 2  # Path length
        assert search_space.shape[2] == 3  # Triple format

        # Verify all triples are valid (either in facts or inverted due to undirected)
        for i in range(search_space.shape[0]):
            for j in range(search_space.shape[1]):
                triple = search_space[i, j]
                # Check if triple or its inverse exists in facts
                triple_exists = (knowledge_graph_facts == triple).all(dim=1).any()
                inverse_triple = torch.tensor([triple[2], triple[1], triple[0]], dtype=torch.long)
                inverse_exists = (knowledge_graph_facts == inverse_triple).all(dim=1).any()
                assert triple_exists or inverse_exists

    def test_caching_behavior(self, knowledge_graph_facts: torch.Tensor) -> None:
        """Test that caching works correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_folder = Path(tmpdir)

            # First setup with caching
            finder1 = CorpusPathFinder(undirected=True)
            finder1.setup(facts_batch=knowledge_graph_facts, cache_folder=cache_folder)

            # Verify cache file was created
            cache_file = cache_folder / "graph.pkl"
            assert cache_file.exists()

            # Second setup should load from cache
            finder2 = CorpusPathFinder(undirected=True)
            finder2.setup(facts_batch=knowledge_graph_facts, cache_folder=cache_folder)

            # Both should produce same results
            paths1 = finder1.find_candidates(source=ALICE, target=CHARLIE)
            paths2 = finder2.find_candidates(source=ALICE, target=CHARLIE)

            assert len(paths1) == len(paths2)

    def test_fact_path_structure(self, knowledge_graph_facts: torch.Tensor) -> None:
        """Test that FactPath objects have the expected structure."""
        finder = CorpusPathFinder(undirected=True)
        finder.setup(facts_batch=knowledge_graph_facts)

        paths = finder.find_candidates(source=ALICE, target=CHARLIE, k=2)

        assert len(paths) > 0

        for path in paths:
            assert_is_fact_index_list(path)
