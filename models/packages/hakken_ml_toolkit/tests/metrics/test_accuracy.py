from collections.abc import Generator

import pytest
import torch

from hakken_ml_toolkit.metrics.core.create_tensors import TensorCreator
from hakken_ml_toolkit.metrics.core.exceptions import NoSamplesError
from hakken_ml_toolkit.metrics.impl.accuracy import Accuracy, AccuracyConfig


@pytest.fixture
def accuracy_metric() -> Generator[Accuracy, None, None]:
    config = AccuracyConfig()
    yield Accuracy(config=config)


def test_accuracy_init(accuracy_metric: Accuracy) -> None:
    assert accuracy_metric._correct == 0
    assert accuracy_metric._total == 0
    assert isinstance(accuracy_metric.config, AccuracyConfig)


def test_accuracy_reset(accuracy_metric: Accuracy) -> None:
    accuracy_metric._correct = 5
    accuracy_metric._total = 10

    accuracy_metric.reset()

    assert accuracy_metric._correct == 0
    assert accuracy_metric._total == 0


def test_accuracy_update(accuracy_metric: Accuracy) -> None:
    scores = TensorCreator.float_tensor(
        [
            [0.1, 0.9],  # Predicted class 1
            [0.8, 0.2],  # Predicted class 0
            [0.3, 0.7],  # Predicted class 1
            [0.6, 0.4],  # Predicted class 0
        ],
    )
    targets = TensorCreator.long_tensor([1, 0, 1, 0])

    accuracy_metric.update(scores=scores, targets=targets)

    assert accuracy_metric._correct == 4
    assert accuracy_metric._total == 4


def test_accuracy_compute(accuracy_metric: Accuracy) -> None:
    accuracy_metric._correct = 8
    accuracy_metric._total = 10

    accuracy_value = accuracy_metric.compute()

    expected_accuracy = 8 / 10
    assert torch.isclose(accuracy_value[0], torch.tensor(expected_accuracy), atol=1e-5)


def test_accuracy_compute_no_samples(accuracy_metric: Accuracy) -> None:
    with pytest.raises(NoSamplesError):
        accuracy_metric.compute()
