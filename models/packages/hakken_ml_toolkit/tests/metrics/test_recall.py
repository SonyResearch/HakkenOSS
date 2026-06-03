from collections.abc import Generator

import pytest
import torch

from hakken_ml_toolkit.metrics.core.exceptions import NoSamplesError
from hakken_ml_toolkit.metrics.impl.recall import Recall, RecallConfig


@pytest.fixture
def recall_metric() -> Generator[Recall, None, None]:
    config = RecallConfig(num_classes=3, average="macro")
    yield Recall(config=config)


def test_recall_init(recall_metric: Recall) -> None:
    assert recall_metric.total_samples == 0
    assert isinstance(recall_metric.config, RecallConfig)
    assert recall_metric.num_classes == 3
    assert recall_metric.average == "macro"
    assert recall_metric.true_positives.shape[0] == 3
    assert recall_metric.false_negatives.shape[0] == 3


def test_recall_reset(recall_metric: Recall) -> None:
    recall_metric.true_positives += torch.tensor([1.0, 2.0, 3.0])
    recall_metric.false_negatives += torch.tensor([0.5, 1.0, 1.5])
    recall_metric.total_samples = 10

    recall_metric.reset()

    assert torch.all(recall_metric.true_positives == 0)
    assert torch.all(recall_metric.false_negatives == 0)
    assert recall_metric.total_samples == 0


def test_recall_update(recall_metric: Recall) -> None:
    scores = torch.tensor(
        [
            [0.8, 0.1, 0.1],  # Predicted class 0
            [0.2, 0.7, 0.1],  # Predicted class 1
            [0.1, 0.2, 0.7],  # Predicted class 2
            [0.6, 0.3, 0.1],  # Predicted class 0
            [0.1, 0.8, 0.1],  # Predicted class 1
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([0, 1, 2, 1, 0], dtype=torch.long)

    recall_metric.update(scores=scores, targets=targets)

    expected_tp = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
    expected_fn = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64)

    assert torch.all(recall_metric.true_positives == expected_tp)
    assert torch.all(recall_metric.false_negatives == expected_fn)
    assert recall_metric.total_samples == 5


def test_recall_compute(recall_metric: Recall) -> None:
    recall_metric.true_positives = torch.tensor([5.0, 3.0, 2.0])
    recall_metric.false_negatives = torch.tensor([2.0, 1.0, 1.0])
    recall_metric.total_samples = 20

    recall_value = recall_metric.compute()

    recall = recall_metric.true_positives / (
        recall_metric.true_positives + recall_metric.false_negatives
    )
    expected_recall = recall.mean().item()

    assert torch.isclose(recall_value[0], torch.tensor(expected_recall), atol=1e-5)


def test_recall_compute_no_samples(recall_metric: Recall) -> None:
    with pytest.raises(NoSamplesError):
        recall_metric.compute()


def test_recall_none_average(recall_metric: Recall) -> None:
    recall_metric.average = "none"

    recall_metric.true_positives = torch.tensor([2.0, 0.0, 1.0])
    recall_metric.false_negatives = torch.tensor([1.0, 2.0, 0.0])
    recall_metric.total_samples = 5

    recall_value = recall_metric.compute()

    expected_recall_per_class = torch.tensor([0.6667, 0.0, 1.0], dtype=torch.float32)

    assert torch.allclose(recall_value, expected_recall_per_class, atol=1e-4)
