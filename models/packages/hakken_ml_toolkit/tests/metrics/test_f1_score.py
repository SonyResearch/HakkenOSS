from collections.abc import Generator

import pytest
import torch

from hakken_ml_toolkit.metrics.core.exceptions import NoSamplesError
from hakken_ml_toolkit.metrics.impl.f1_score import F1, F1Config


@pytest.fixture
def f1_metric() -> Generator[F1, None, None]:
    config = F1Config(num_classes=3, average="macro")
    yield F1(config=config)


def test_f1_init(f1_metric: F1) -> None:
    assert f1_metric._total == 0
    assert isinstance(f1_metric.config, F1Config)
    assert f1_metric.config.num_classes == 3
    assert f1_metric.config.average == "macro"
    assert f1_metric._true_positives.shape[0] == 3
    assert f1_metric._false_positives.shape[0] == 3
    assert f1_metric._false_negatives.shape[0] == 3


def test_f1_reset(f1_metric: F1) -> None:
    f1_metric._true_positives += torch.tensor([1.0, 2.0, 3.0])
    f1_metric._false_positives += torch.tensor([0.5, 1.0, 1.5])
    f1_metric._false_negatives += torch.tensor([0.2, 0.4, 0.6])
    f1_metric._total = 10

    f1_metric.reset()

    assert torch.all(f1_metric._true_positives == 0)
    assert torch.all(f1_metric._false_positives == 0)
    assert torch.all(f1_metric._false_negatives == 0)
    assert f1_metric._total == 0


def test_f1_update(f1_metric: F1) -> None:
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

    # Call update with the mocked inputs
    f1_metric.update(scores=scores, targets=targets)

    expected_tp = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
    expected_fp = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64)
    expected_fn = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64)

    assert torch.all(f1_metric._true_positives == expected_tp)
    assert torch.all(f1_metric._false_positives == expected_fp)
    assert torch.all(f1_metric._false_negatives == expected_fn)
    assert f1_metric._total == 5


def test_f1_compute(f1_metric: F1) -> None:
    f1_metric._true_positives = torch.tensor([5.0, 3.0, 2.0])
    f1_metric._false_positives = torch.tensor([2.0, 1.0, 1.0])
    f1_metric._false_negatives = torch.tensor([1.0, 2.0, 3.0])
    f1_metric._total = 20

    f1_value = f1_metric.compute()

    precision = f1_metric._true_positives / (f1_metric._true_positives + f1_metric._false_positives)
    recall = f1_metric._true_positives / (f1_metric._true_positives + f1_metric._false_negatives)
    f1_per_class = 2 * (precision * recall) / (precision + recall)

    expected_f1 = f1_per_class.mean().item()

    assert torch.isclose(f1_value[0], torch.tensor(expected_f1), atol=1e-5)


def test_f1_compute_no_samples(f1_metric: F1) -> None:
    with pytest.raises(NoSamplesError):
        f1_metric.compute()


def test_f1_none_average(f1_metric: F1) -> None:
    f1_metric.config.average = "none"

    f1_metric._true_positives = torch.tensor([2.0, 0.0, 1.0])
    f1_metric._false_positives = torch.tensor([1.0, 0.0, 1.0])
    f1_metric._false_negatives = torch.tensor([1.0, 2.0, 0.0])
    f1_metric._total = 5

    f1_value = f1_metric.compute()

    expected_f1_per_class = torch.tensor([0.6667, 0.0, 0.6667], dtype=torch.float32)

    assert torch.allclose(f1_value, expected_f1_per_class, atol=1e-4)
