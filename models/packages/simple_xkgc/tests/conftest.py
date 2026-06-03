"""Common test fixtures and mocks for hakken_explainer tests."""

import pytest
import torch
from kge.common.entities import KGPredictionSubgraph
from torch import Tensor, nn


class MockGNNKGE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.counter = -5

    def reset(self) -> None:
        self.counter = -5

    def score(self, pred_subgraph: KGPredictionSubgraph) -> torch.Tensor:
        # Mock scoring function: returns decreasing scores for necessity testing
        num_edges = pred_subgraph.edge_label_index.shape[1]
        values = []
        for _ in range(num_edges):
            self.counter += 1
            # Higher scores when more context is removed (necessity effect)
            values.append(self.counter * 0.1)
        return torch.tensor(values).unsqueeze(1)


@pytest.fixture
def mock_model() -> MockGNNKGE:
    """Create a mock GNNKGE model for testing."""
    return MockGNNKGE()


@pytest.fixture
def sample_context_kg() -> Tensor:
    """Create a small sample knowledge graph for testing."""
    # Simple KG: entities 0-5, relations 10-12
    return torch.tensor(
        [
            [0, 10, 1],
            [2, 11, 0],
            [3, 12, 2],
            [5, 11, 3],
            [3, 10, 4],
            [4, 12, 5],
            [4, 10, 1],
            [1, 12, 3],
            [2, 10, 4],
        ]
    )


BATCH_SIZE = 2
FACTS_NUM_DIMS = 2  # [num_facts, 3]
FACTS_NUM_COLS = 3  # subject, relation, object


def assert_valid_scores(scores: list[float], expected_length: int) -> None:
    """Assert that scores are valid.

    Args:
        scores: List of scores to validate.
        expected_length: Expected number of scores.
    """
    assert isinstance(scores, list)
    assert len(scores) == expected_length
    assert all(isinstance(score, float) for score in scores)
    assert all(not torch.isnan(torch.tensor(score)) for score in scores)
