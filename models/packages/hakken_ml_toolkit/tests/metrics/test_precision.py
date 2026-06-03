from collections.abc import Generator

import pytest
import torch

from hakken_ml_toolkit.metrics.core.exceptions import NoSamplesError
from hakken_ml_toolkit.metrics.impl.precision import Precision, PrecisionConfig


@pytest.fixture
def precision_metric() -> Generator[Precision, None, None]:
    config: PrecisionConfig = PrecisionConfig(num_classes=3, average="macro")
    yield Precision(config=config)


def test_precision_init(precision_metric: Precision) -> None:
    assert precision_metric.total_samples == 0
    assert isinstance(precision_metric.config, PrecisionConfig)
    assert precision_metric.config.num_classes == 3
    assert precision_metric.config.average == "macro"
    assert precision_metric.true_positives.shape[0] == 3
    assert precision_metric.false_positives.shape[0] == 3


def test_precision_reset(precision_metric: Precision) -> None:
    precision_metric.true_positives += torch.tensor([1.0, 2.0, 3.0])
    precision_metric.false_positives += torch.tensor([0.5, 1.0, 1.5])
    precision_metric.total_samples = 10

    precision_metric.reset()

    assert torch.all(precision_metric.true_positives == 0)
    assert torch.all(precision_metric.false_positives == 0)
    assert precision_metric.total_samples == 0


def test_precision_update(precision_metric: Precision) -> None:
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

    precision_metric.update(scores=scores, targets=targets)

    expected_tp = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
    expected_fp = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64)

    assert torch.all(precision_metric.true_positives == expected_tp)
    assert torch.all(precision_metric.false_positives == expected_fp)
    assert precision_metric.total_samples == 5


def test_precision_compute(precision_metric: Precision) -> None:
    precision_metric.true_positives = torch.tensor([5.0, 3.0, 2.0])
    precision_metric.false_positives = torch.tensor([2.0, 1.0, 1.0])
    precision_metric.total_samples = 20

    precision_value = precision_metric.compute()

    precision = precision_metric.true_positives / (
        precision_metric.true_positives + precision_metric.false_positives
    )
    expected_precision = precision.mean().item()

    assert torch.isclose(precision_value[0], torch.tensor(expected_precision), atol=1e-5)


def test_precision_compute_no_samples(precision_metric: Precision) -> None:
    with pytest.raises(NoSamplesError):
        precision_metric.compute()


def test_precision_none_average(precision_metric: Precision) -> None:
    precision_metric.config.average = "none"

    precision_metric.true_positives = torch.tensor([2.0, 0.0, 1.0])
    precision_metric.false_positives = torch.tensor([1.0, 0.0, 1.0])
    precision_metric.total_samples = 5

    precision_value = precision_metric.compute()

    expected_precision_per_class = torch.tensor([0.6667, 0.0, 0.5], dtype=torch.float32)

    assert torch.allclose(precision_value, expected_precision_per_class, atol=1e-4)
