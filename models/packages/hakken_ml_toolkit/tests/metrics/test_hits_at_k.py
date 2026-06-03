from collections.abc import Generator

import pytest
import torch

from hakken_ml_toolkit.metrics.core.exceptions import NoSamplesError
from hakken_ml_toolkit.metrics.impl.hits_at_k import HitsAtK, HitsAtKConfig


@pytest.fixture
def hits_at_k() -> Generator[HitsAtK, None, None]:
    config = HitsAtKConfig(top_k=3)
    yield HitsAtK(config=config)


def test_hits_at_k_init(hits_at_k: HitsAtK) -> None:
    assert hits_at_k._total_hits == 0
    assert hits_at_k._total == 0
    assert isinstance(hits_at_k.config, HitsAtKConfig)


def test_hits_at_k_reset(hits_at_k: HitsAtK) -> None:
    hits_at_k._total_hits = 5
    hits_at_k._total = 10

    hits_at_k.reset()

    assert hits_at_k._total_hits == 0
    assert hits_at_k._total == 0


def test_hits_at_k_update(hits_at_k: HitsAtK) -> None:
    scores = torch.tensor([[0.1, 0.4, 0.5], [0.2, 0.6, 0.2], [0.7, 0.2, 0.1]])
    targets = torch.tensor([2, 1, 0])

    hits_at_k.update(scores=scores, targets=targets)

    assert hits_at_k._total_hits == 3  # All targets are within the top 3 scores
    assert hits_at_k._total == 3


def test_hits_at_k_compute(hits_at_k: HitsAtK) -> None:
    hits_at_k._total_hits = 3
    hits_at_k._total = 5

    hits_at_k_value = hits_at_k.compute()

    expected_hits_at_k = 3 / 5
    assert torch.isclose(hits_at_k_value[0], torch.tensor(expected_hits_at_k))


def test_hits_at_k_compute_no_samples(hits_at_k: HitsAtK) -> None:
    with pytest.raises(NoSamplesError):
        hits_at_k.compute()
