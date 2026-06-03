import pytest

try:
    import torch
except ImportError:
    pytest.skip("PyTorch is not installed", allow_module_level=True)


import networkx as nx

from hakken_ml_toolkit.ml_utils.extras import FactBatchUtils
from hakken_ml_toolkit.ml_utils.extras.domain import ProximityNetworkData


@pytest.fixture
def sample_fact_batch() -> torch.Tensor:
    """
    Creates a sample fact batch for testing.
    Format: [subject, relation, object, timestamp]
    """
    return torch.tensor(
        [
            [0, 0, 1, 100],  # Entity 0 has relation 0 with entity 1 at time 100
            [1, 1, 2, 200],  # Entity 1 has relation 1 with entity 2 at time 200
            [2, 0, 3, 300],  # Entity 2 has relation 0 with entity 3 at time 300
            [3, 2, 0, 400],  # Entity 3 has relation 2 with entity 0 at time 400
            [0, 1, 3, 500],  # Entity 0 has relation 1 with entity 3 at time 500
        ],
        dtype=torch.long,
    )


@pytest.fixture
def sample_so_batch() -> torch.Tensor:
    return torch.tensor(
        [
            [0, 1],  # Subject 0, Object 1
            [1, 2],  # Subject 1, Object 2
            [2, 3],  # Subject 2, Object 3
        ],
        dtype=torch.long,
    )


@pytest.fixture
def sample_ro_batch() -> torch.Tensor:
    return torch.tensor(
        [
            [0, 1],  # Relation 0, Object 1
            [1, 2],  # Relation 1, Object 2
            [2, 3],  # Relation 2, Object 3
        ],
        dtype=torch.long,
    )


@pytest.fixture
def sample_sro_batch() -> torch.Tensor:
    return torch.tensor(
        [
            [0, 0, 1],  # Subject 0, Relation 0, Object 1
            [1, 1, 2],  # Subject 1, Relation 1, Object 2
            [2, 0, 3],  # Subject 2, Relation 0, Object 3
            [3, 2, 0],  # Subject 3, Relation 2, Object 0
        ],
        dtype=torch.long,
    )


class TestFactBatchUtils:
    def test_num_entities(self, sample_fact_batch: torch.Tensor) -> None:
        num_entities = FactBatchUtils.num_entities(sample_fact_batch)
        assert num_entities == 4

    def test_num_relations(self, sample_fact_batch: torch.Tensor) -> None:
        num_relations = FactBatchUtils.num_relations(sample_fact_batch)
        assert num_relations == 3

    def test_subject(self, sample_fact_batch: torch.Tensor) -> None:
        subjects = FactBatchUtils.subject(sample_fact_batch)
        expected = torch.tensor([0, 1, 2, 3, 0], dtype=torch.long)
        assert torch.all(subjects == expected)

    def test_relation(self, sample_fact_batch: torch.Tensor) -> None:
        relations = FactBatchUtils.relation(sample_fact_batch)
        expected = torch.tensor([0, 1, 0, 2, 1], dtype=torch.long)
        assert torch.all(relations == expected)

    def test_object(self, sample_fact_batch: torch.Tensor) -> None:
        objects = FactBatchUtils.object(sample_fact_batch)
        expected = torch.tensor([1, 2, 3, 0, 3], dtype=torch.long)
        assert torch.all(objects == expected)

    def test_timestamp(self, sample_fact_batch: torch.Tensor) -> None:
        timestamps = FactBatchUtils.timestamp(sample_fact_batch)
        expected = torch.tensor([100, 200, 300, 400, 500], dtype=torch.long)
        assert torch.all(timestamps == expected)

    def test_to_so_batch(self, sample_fact_batch: torch.Tensor) -> None:
        so_batch = FactBatchUtils.to_so_batch(sample_fact_batch)
        expected = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0], [0, 3]], dtype=torch.long)
        assert torch.all(so_batch == expected)

    def test_to_sr_batch(self, sample_fact_batch: torch.Tensor) -> None:
        sr_batch = FactBatchUtils.to_sr_batch(sample_fact_batch)
        expected = torch.tensor([[0, 0], [1, 1], [2, 0], [3, 2], [0, 1]], dtype=torch.long)
        assert torch.all(sr_batch == expected)

    def test_to_ro_batch(self, sample_fact_batch: torch.Tensor) -> None:
        ro_batch = FactBatchUtils.to_ro_batch(sample_fact_batch)
        expected = torch.tensor([[0, 1], [1, 2], [0, 3], [2, 0], [1, 3]], dtype=torch.long)
        assert torch.all(ro_batch == expected)

    def test_to_so_batch_and_relations(self, sample_fact_batch: torch.Tensor) -> None:
        """Test converting a fact batch to unique subject-object pairs and associated relations."""
        # Test with provided num_relations
        so_unique_batch, target = FactBatchUtils.to_so_batch_and_relations(
            sample_fact_batch, num_relations=3
        )

        # Check so_unique_batch - should be unique S-O pairs
        expected_so = torch.tensor([[0, 1], [0, 3], [1, 2], [2, 3], [3, 0]], dtype=torch.long)

        assert torch.all(so_unique_batch == expected_so)

        # Check target - should be a one-hot encoding of relations for each unique S-O pair
        expected_target = torch.zeros((5, 3), dtype=torch.long)
        expected_target[0, 0] = 1  # (0,1) has relation 0
        expected_target[1, 1] = 1  # (0,3) has relation 1
        expected_target[2, 1] = 1  # (1,2) has relation 1
        expected_target[3, 0] = 1  # (2,3) has relation 0
        expected_target[4, 2] = 1  # (3,0) has relation 2

        assert torch.all(target == expected_target)

        # Test without provided num_relations
        so_unique_batch2, target2 = FactBatchUtils.to_so_batch_and_relations(sample_fact_batch)
        assert torch.all(so_unique_batch2 == expected_so)
        assert torch.all(target2 == expected_target)

    def test_fact_batch_pair_relation_labels_aligns_with_facts(
        self, sample_fact_batch: torch.Tensor
    ) -> None:
        """Per-fact rows equal target[inverse] from the same unique (s,o) pass."""
        per_fact = FactBatchUtils.fact_batch_pair_relation_labels(
            sample_fact_batch, num_relations=3
        )
        assert per_fact.shape == (sample_fact_batch.shape[0], 3)
        assert per_fact.dtype == torch.float32
        _, target_unique = FactBatchUtils.to_so_batch_and_relations(
            sample_fact_batch, num_relations=3
        )
        so = FactBatchUtils.to_so_batch(sample_fact_batch[:, :3])
        _, inv = torch.unique(so, dim=0, return_inverse=True)
        torch.testing.assert_close(per_fact, target_unique[inv].to(dtype=torch.float32))

    def test_so_to_sro_batch(self, sample_so_batch: torch.Tensor) -> None:
        """Test converting a subject-object batch to a subject-relation-object batch."""
        num_relations = 2
        sro_batch = FactBatchUtils.so_to_sro_batch(sample_so_batch, num_relations)

        # Should have num_relations * sample_so_batch.size(0) rows
        expected_size = (num_relations * sample_so_batch.size(0), 3)
        assert sro_batch.size() == expected_size

        # Check some specific values
        # For first S-O pair (0,1) we should have all relations
        assert torch.all(sro_batch[0] == torch.tensor([0, 0, 1]))
        assert torch.all(sro_batch[1] == torch.tensor([0, 1, 1]))

        # For second S-O pair (1,2) we should have all relations
        assert torch.all(sro_batch[2] == torch.tensor([1, 0, 2]))
        assert torch.all(sro_batch[3] == torch.tensor([1, 1, 2]))

    def test_ro_to_sro_batch(self, sample_ro_batch: torch.Tensor) -> None:
        """Test converting a relation-object batch to a subject-relation-object batch."""
        num_entities = 3
        sro_batch = FactBatchUtils.ro_to_sro_batch(sample_ro_batch, num_entities)

        # Should have num_entities * sample_ro_batch.size(0) rows
        expected_size = (num_entities * sample_ro_batch.size(0), 3)
        assert sro_batch.size() == expected_size

        # Check some specific values
        # For first R-O pair (0,1) we should have all subjects
        assert torch.all(sro_batch[0] == torch.tensor([0, 0, 1]))
        assert torch.all(sro_batch[1] == torch.tensor([1, 0, 1]))
        assert torch.all(sro_batch[2] == torch.tensor([2, 0, 1]))

        # For second R-O pair (1,2) we should have all subjects
        assert torch.all(sro_batch[3] == torch.tensor([0, 1, 2]))

    def test_generate_fact_batch_proximity_graph(self, sample_sro_batch: torch.Tensor) -> None:
        """Test generating a proximity graph from a fact batch."""
        num_entities = 4
        max_distance = 2
        neighbors_seq = FactBatchUtils.generate_fact_batch_proximity_graph(
            sample_sro_batch, num_entities, max_distance
        )

        # Check return type
        assert isinstance(neighbors_seq, ProximityNetworkData)

        # Neighbors for entity 0 should include 1 (direct) and 3 (direct)
        # Distance 1: entity 0 is connected to entity 1 and entity 3 is connected to 0
        entity_0_neighbors = neighbors_seq.neighbors[0]
        entity_0_distances = neighbors_seq.distances[0]

        # Find non-padding values
        valid_indices = entity_0_neighbors >= 0
        valid_neighbors = entity_0_neighbors[valid_indices]

        # Check entity 0's neighbors
        assert 1 in valid_neighbors
        assert 3 in valid_neighbors

        # We also expect entity 2 might be distance 2 from entity 0 (through entity 1)
        # And entity 3 is distance 1 from entity 0

        # Check that all distances are valid (1 or 2)
        valid_distances = entity_0_distances[valid_indices]
        assert torch.all((valid_distances == 1) | (valid_distances == 2))

    def test_to_networkx(self, sample_sro_batch: torch.Tensor) -> None:
        """Test converting a fact batch to a NetworkX graph."""
        num_entities = 4
        num_relations = 3

        # Test with provided num_entities and num_relations
        graph = FactBatchUtils.to_networkx(
            sample_sro_batch, num_entities=num_entities, num_relations=num_relations
        )

        # Check graph type and properties
        assert isinstance(graph, nx.MultiDiGraph)
        assert graph.number_of_nodes() == num_entities
        assert graph.graph["num_relations"] == num_relations

        # Check edges
        assert graph.has_edge(0, 1)  # Subject 0 to Object 1
        assert graph.has_edge(1, 2)  # Subject 1 to Object 2
        assert graph.has_edge(2, 3)  # Subject 2 to Object 3
        assert graph.has_edge(3, 0)  # Subject 3 to Object 0

        # Check relation attributes
        edge_data = graph.get_edge_data(0, 1, 0)  # Get data for first edge
        assert edge_data["relation"] == 0

        # Test without provided num_entities and num_relations
        graph2 = FactBatchUtils.to_networkx(sample_sro_batch)
        assert isinstance(graph2, nx.MultiDiGraph)
        assert graph2.graph["num_relations"] == 3  # Max relation + 1

    def test_remove_batch(self, sample_sro_batch: torch.Tensor) -> None:
        """Test removing triples from a fact batch."""
        # Create a batch to remove
        batch_to_remove = torch.tensor(
            [
                [0, 0, 1],  # Remove the first triple
                [2, 0, 3],  # Remove the third triple
            ],
            dtype=torch.long,
        )

        # Remove the batch
        result = FactBatchUtils.remove_batch(sample_sro_batch, batch_to_remove)

        # Expected result: original batch minus the removed triples
        expected = torch.tensor(
            [
                [1, 1, 2],  # Second triple
                [3, 2, 0],  # Fourth triple
            ],
            dtype=torch.long,
        )

        assert torch.all(result == expected)
        assert result.shape[0] == 2  # Two triples left
