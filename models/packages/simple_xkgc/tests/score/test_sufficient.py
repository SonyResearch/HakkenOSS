import random
from typing import cast

import pytest
import torch
from datasets.common.constants import DataSplits
from hakken_ml_toolkit.ml_base_structures.data_generator import DummyDataGenerator
from kge.models.gnn import GNNKGE
from kge.scores import ComplExScore
from torch import Tensor
from torch_geometric.nn.models import GraphSAGE

from hakken_explainer.scores.sufficient import SufficientScore
from tests.conftest import (
    BATCH_SIZE,
    FACTS_NUM_COLS,
    FACTS_NUM_DIMS,
    MockGNNKGE,
    assert_valid_scores,
)


class TestSufficientScore:
    @pytest.fixture
    def mock_model(self) -> MockGNNKGE:
        """Create a mock GNNKGE model for testing."""
        return MockGNNKGE()

    @pytest.fixture
    def sample_context_kg(self) -> Tensor:
        """Create a small sample knowledge graph for testing."""
        # Simple KG: entities 0-5, relations 10-12 0 1 2 3 4
        return torch.tensor(
            [
                [0, 10, 1],
                [2, 11, 0],
                [3, 12, 2],
                [5, 11, 3],
                [3, 10, 4],
                [4, 12, 5],
                [4, 10, 1],
            ]
        )

    @pytest.fixture
    def scorer(self, sample_context_kg: Tensor, mock_model: MockGNNKGE) -> SufficientScore:
        """Create SufficientScore instance for testing."""
        return SufficientScore(sample_context_kg, cast("GNNKGE", mock_model))

    def test_initialization(self, sample_context_kg: Tensor, mock_model: MockGNNKGE) -> None:
        """Test that SufficientScore initializes correctly."""
        scorer: SufficientScore = SufficientScore(sample_context_kg, cast("GNNKGE", mock_model))

        assert torch.equal(scorer.context_kg, sample_context_kg)
        assert scorer.model == mock_model

    @pytest.mark.parametrize(
        "k, expected_num_facts",
        [
            (1, 2),
            (2, 4),
        ],
    )
    def test_get_context_kg_basic_functionality(
        self, scorer: SufficientScore, k: int, expected_num_facts: int
    ) -> None:
        """Test basic functionality of get_context_kg method."""

        target_fact = torch.tensor([[0, 10, 1]])

        context_subgraph: Tensor = scorer.get_context_kg(target_fact, k=k)

        assert context_subgraph.dim() == FACTS_NUM_DIMS
        assert context_subgraph.shape[1] == FACTS_NUM_COLS

        assert context_subgraph.shape[0] == expected_num_facts

    def test_get_context_kg_preserves_original_ids(self, scorer: SufficientScore) -> None:
        """Test that get_context_kg preserves original entity IDs."""
        target_fact: Tensor = torch.tensor([[0, 10, 1]])

        context_subgraph: Tensor = scorer.get_context_kg(target_fact, k=1)
        assert context_subgraph.dim() == FACTS_NUM_DIMS
        assert context_subgraph.shape[1] == FACTS_NUM_COLS

        unique_entities: Tensor = torch.unique(context_subgraph[:, [0, 2]])
        expected_entities: set[int] = {0, 1, 2, 4}
        assert all(entity.item() in expected_entities for entity in unique_entities)

    @pytest.mark.parametrize("normalize_by_original", [True, False])
    def test_score_basic_functionality(
        self, scorer: SufficientScore, normalize_by_original: bool
    ) -> None:
        """Test basic functionality of score method."""
        target_fact: Tensor = torch.tensor([[0, 10, 1]])
        candidate_paths: Tensor = torch.tensor(
            [
                [[0, 10, 2], [1, 11, 2]],
                [[0, 11, 3], [3, 10, 4]],
            ]
        )

        scores = scorer.score(
            target_fact,
            candidate_paths,
            batch_size=BATCH_SIZE,
            normalize_by_original=normalize_by_original,
        )

        # Should return a list of scores
        assert isinstance(scores, list)
        assert len(scores) == candidate_paths.shape[0]  # One score per path
        assert all(isinstance(score, float) for score in scores)

    def test_score_single_path(self, scorer: SufficientScore) -> None:
        """Test scoring with a single explanation path."""
        target_fact: Tensor = torch.tensor([[0, 10, 1]])
        candidate_paths: Tensor = torch.tensor([[[0, 10, 4], [1, 11, 2]]])

        scores: list[float] = scorer.score(target_fact, candidate_paths, batch_size=1)

        assert len(scores) == 1
        assert isinstance(scores[0], float)

    def test_score_empty_paths(self, scorer: SufficientScore) -> None:
        """Test scoring with empty candidate paths."""
        target_fact: Tensor = torch.tensor([[0, 10, 1]])
        candidate_paths: Tensor = torch.empty((0, 2, 3))  # Empty tensor

        scores: list[float] = scorer.score(target_fact, candidate_paths)

        assert len(scores) == 0

    def test_score_different_batch_sizes(self, scorer: SufficientScore) -> None:
        """Test that different batch sizes produce same results."""
        target_fact: Tensor = torch.tensor([[0, 10, 1]])
        candidate_paths: Tensor = torch.tensor(
            [
                [[0, 10, 2], [1, 11, 2]],
                [[0, 11, 3], [3, 10, 4]],
                [[1, 10, 4], [4, 12, 5]],
            ]
        )

        model = cast("MockGNNKGE", scorer.model)

        model.reset()

        scores_batch1: list[float] = scorer.score(target_fact, candidate_paths, batch_size=1)

        model.reset()

        scores_batch3: list[float] = scorer.score(target_fact, candidate_paths, batch_size=3)

        # Results should be identical regardless of batch size
        assert len(scores_batch1) == len(scores_batch3)
        for s1, s3 in zip(scores_batch1, scores_batch3, strict=False):
            assert s1 == pytest.approx(s3, rel=1e-6)

    def test_score_tensor_shapes(self, scorer: SufficientScore) -> None:
        """Test that score method handles tensor shapes correctly."""
        target_fact: Tensor = torch.tensor([[0, 10, 1]])

        candidate_paths: Tensor = torch.tensor(
            [
                [[0, 10, 1], [1, 11, 2], [2, 12, 3]],  # 3-hop path
            ]
        )

        scores: list[float] = scorer.score(target_fact, candidate_paths)
        assert len(scores) == 1

    def test_device_consistency(self, scorer: SufficientScore) -> None:
        """Test that the scorer handles device placement correctly."""
        target_fact: Tensor = torch.tensor([[0, 10, 1]])
        candidate_paths: Tensor = torch.tensor([[[0, 10, 1], [1, 11, 2]]])

        scores = scorer.score(target_fact, candidate_paths)
        assert len(scores) == 1

        if torch.cuda.is_available():
            target_fact_gpu = target_fact.cuda()
            candidate_paths_gpu = candidate_paths.cuda()

            scores_gpu = scorer.score(target_fact_gpu, candidate_paths_gpu)
            assert len(scores_gpu) == 1

    def test_score_batch_size_parameter(self, scorer: SufficientScore) -> None:
        """Test score method with different batch sizes."""
        target_fact: Tensor = torch.tensor([[0, 10, 1]])
        candidate_paths: Tensor = torch.tensor(
            [
                [[0, 10, 1], [1, 11, 2]],
                [[0, 11, 3], [3, 10, 4]],
            ]
        )

        # Test with batch_size=1
        scores_small_batch: list[float] = scorer.score(target_fact, candidate_paths, batch_size=1)

        # Test with batch_size larger than number of paths
        scores_large_batch: list[float] = scorer.score(target_fact, candidate_paths, batch_size=10)

        assert len(scores_small_batch) == len(scores_large_batch) == candidate_paths.shape[0]


# Additional integration test (optional, requires actual model)
class TestSufficientScoreIntegration:
    """Integration tests that would require actual model and data."""

    @pytest.mark.parametrize("seed", [0, 1, 42, 1234])
    def test_with_real_model(self, seed: int) -> None:
        embedding_dim = random.choice([16, 32, 64])
        num_layers = random.randint(1, 3)

        knowledge_graph = DummyDataGenerator.knowledge_graph_from_seed(
            seed=seed, batch_size_values=[100, 500, 1000]
        )
        gnn = GraphSAGE(
            in_channels=embedding_dim,
            hidden_channels=embedding_dim,
            num_layers=num_layers,
        )
        score_fn = ComplExScore()

        model = GNNKGE(
            embedding_dim=embedding_dim,
            num_entities=cast("int", knowledge_graph.num_entities),
            num_relations=cast("int", knowledge_graph.num_relations),
            gnn=gnn,
            score_fn=score_fn,
        )

        context_kg = knowledge_graph.facts_dict[DataSplits.TRAIN]

        scorer = SufficientScore(context_kg=context_kg, model=model)

        target_fact = torch.tensor([[0, 0, 1]])

        candidate_explanations: list[Tensor] = []
        num_explanations = random.randint(1, 10)
        explanation_length = random.randint(2, 6)

        for n in range(num_explanations):
            path = []
            for step in range(explanation_length):
                idx = n * explanation_length + step
                path.append(context_kg[idx])

            candidate_explanations.append(torch.stack(path))

        candidate_paths = torch.stack(candidate_explanations)
        scores = scorer.score(
            target_fact=target_fact,
            candidate_paths=candidate_paths,
            batch_size=2,
        )
        assert_valid_scores(scores, expected_length=candidate_paths.shape[0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
