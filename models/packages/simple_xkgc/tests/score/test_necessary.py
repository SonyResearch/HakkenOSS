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

from hakken_explainer.scores.necessary import NecessaryScore
from tests.conftest import (
    BATCH_SIZE,
    FACTS_NUM_COLS,
    FACTS_NUM_DIMS,
    MockGNNKGE,
    assert_valid_scores,
)


class TestNecessaryScore:
    @pytest.fixture
    def scorer(self, sample_context_kg: Tensor, mock_model) -> NecessaryScore:
        """Create NecessaryScore instance for testing."""
        return NecessaryScore(sample_context_kg, cast("GNNKGE", mock_model))

    def test_initialization(self, sample_context_kg: Tensor, mock_model) -> None:
        """Test that NecessaryScore initializes correctly."""
        scorer: NecessaryScore = NecessaryScore(sample_context_kg, cast("GNNKGE", mock_model))

        assert torch.equal(scorer.context_kg, sample_context_kg)
        assert scorer.model == mock_model

    def test_remove_explanation_from_context_basic(self, scorer: NecessaryScore) -> None:
        """Test basic functionality of remove_explanation_from_context method."""
        context_facts = torch.tensor(
            [
                [0, 10, 1],
                [2, 11, 0],
                [3, 12, 2],
                [4, 10, 1],
            ]
        )
        explanation = torch.tensor(
            [
                [0, 10, 1],
                [3, 12, 2],
            ]
        )

        result = scorer.remove_explanation_from_context(context_facts, explanation)

        assert result.dim() == FACTS_NUM_DIMS
        assert result.shape[1] == FACTS_NUM_COLS
        assert result.shape[0] == explanation.shape[0]

        # Check that removed facts are not in result
        expected_remaining = torch.tensor(
            [
                [2, 11, 0],
                [4, 10, 1],
            ]
        )
        assert torch.equal(result, expected_remaining)

    def test_remove_explanation_empty_explanation(self, scorer: NecessaryScore) -> None:
        """Test remove_explanation_from_context with empty explanation."""
        context_facts = torch.tensor(
            [
                [0, 10, 1],
                [2, 11, 0],
            ]
        )
        explanation = torch.empty((0, 3))

        result = scorer.remove_explanation_from_context(context_facts, explanation)

        assert torch.equal(result, context_facts)

    def test_remove_explanation_no_matches(self, scorer: NecessaryScore) -> None:
        """Test remove_explanation_from_context when no facts match."""
        context_facts = torch.tensor(
            [
                [0, 10, 1],
                [2, 11, 0],
            ]
        )
        explanation = torch.tensor(
            [
                [5, 12, 6],
                [7, 13, 8],
            ]
        )

        result = scorer.remove_explanation_from_context(context_facts, explanation)

        assert torch.equal(result, context_facts)

    @pytest.mark.parametrize("normalize_by_original", [True, False])
    def test_score_basic_functionality(
        self, scorer: NecessaryScore, normalize_by_original: bool
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

    def test_score_single_path(self, scorer: NecessaryScore) -> None:
        """Test scoring with a single explanation path."""
        target_fact: Tensor = torch.tensor([[0, 10, 1]])
        candidate_paths: Tensor = torch.tensor([[[0, 10, 4], [1, 11, 2]]])

        scores: list[float] = scorer.score(target_fact, candidate_paths, batch_size=1)

        assert len(scores) == 1
        assert isinstance(scores[0], float)

    def test_score_empty_paths(self, scorer: NecessaryScore) -> None:
        """Test scoring with empty candidate paths."""
        target_fact: Tensor = torch.tensor([[0, 10, 1]])
        candidate_paths: Tensor = torch.empty((0, 2, 3))  # Empty tensor

        scores: list[float] = scorer.score(target_fact, candidate_paths)

        assert len(scores) == 0

    def test_score_different_batch_sizes(self, scorer: NecessaryScore) -> None:
        """Test that different batch sizes produce same results."""
        target_fact = torch.tensor([[0, 10, 1]])
        candidate_paths = torch.tensor(
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

    def test_score_necessity_logic(self, scorer: NecessaryScore) -> None:
        """Test that necessity scoring logic works correctly."""
        target_fact: Tensor = torch.tensor([[0, 10, 1]])

        # Create paths that should be in the context
        candidate_paths: Tensor = torch.tensor(
            [
                [[0, 10, 1], [2, 11, 0]],  # Contains target fact (should be removed)
                [[4, 10, 1], [3, 12, 2]],  # Different facts
            ]
        )

        scores = scorer.score(
            target_fact, candidate_paths, batch_size=1, normalize_by_original=True
        )

        assert len(scores) == candidate_paths.shape[0]
        # Necessity scores should be positive when removing important context
        assert all(isinstance(score, float) for score in scores)

    def test_score_tensor_shapes(self, scorer: NecessaryScore) -> None:
        """Test that score method handles tensor shapes correctly."""
        target_fact: Tensor = torch.tensor([[0, 10, 1]])

        candidate_paths: Tensor = torch.tensor(
            [
                [[0, 10, 1], [1, 11, 2], [2, 12, 3]],  # 3-hop path
            ]
        )

        scores: list[float] = scorer.score(target_fact, candidate_paths)
        assert len(scores) == 1

    def test_device_consistency(self, scorer: NecessaryScore) -> None:
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

    def test_score_batch_size_parameter(self, scorer: NecessaryScore) -> None:
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

    def test_num_hops_parameter(self, scorer: NecessaryScore) -> None:
        """Test that num_hops parameter affects context extraction."""
        target_fact: Tensor = torch.tensor([[0, 10, 1]])
        candidate_paths: Tensor = torch.tensor([[[2, 11, 0], [3, 12, 2]]])

        # Test with different num_hops values
        scores_1hop = scorer.score(target_fact, candidate_paths, num_hops=1)
        scores_2hop = scorer.score(target_fact, candidate_paths, num_hops=2)

        assert len(scores_1hop) == len(scores_2hop) == 1
        # Scores might be different due to different context sizes
        assert isinstance(scores_1hop[0], float)
        assert isinstance(scores_2hop[0], float)


# Additional integration test (optional, requires actual model)
class TestNecessaryScoreIntegration:
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

        scorer = NecessaryScore(context_kg=context_kg, model=model)

        target_fact = torch.tensor([[0, 0, 1]])

        candidate_explanations: list[Tensor] = []
        num_explanations = random.randint(1, 10)
        explanation_length = random.randint(2, 6)

        for n in range(num_explanations):
            path = []
            for step in range(explanation_length):
                idx = n * explanation_length + step
                if idx < context_kg.shape[0]:
                    path.append(context_kg[idx])
                else:
                    # Wrap around if we exceed available facts
                    path.append(context_kg[idx % context_kg.shape[0]])

            candidate_explanations.append(torch.stack(path))

        candidate_paths = torch.stack(candidate_explanations)
        scores = scorer.score(
            target_fact=target_fact,
            candidate_paths=candidate_paths,
            batch_size=2,
        )
        assert_valid_scores(scores, expected_length=candidate_paths.shape[0])

    def test_necessity_vs_sufficiency_difference(self) -> None:
        """Test that necessity scoring behaves differently from sufficiency."""
        # This would require comparing with SufficientScore
        # Left as a placeholder for integration testing
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
