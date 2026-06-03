from collections.abc import Generator

import pytest
import torch

from hakken_ml_toolkit.metrics.core.exceptions import NoSamplesError
from hakken_ml_toolkit.metrics.impl.mean_rank import MeanRank, MeanRankConfig


@pytest.fixture
def mean_rank() -> Generator[MeanRank, None, None]:
    config = MeanRankConfig()
    yield MeanRank(config=config)


def test_mean_rank_init(mean_rank: MeanRank) -> None:
    assert mean_rank.ranks_sum == 0
    assert mean_rank.num_samples == 0
    assert isinstance(mean_rank.config, MeanRankConfig)


def test_mean_rank_reset(mean_rank: MeanRank) -> None:
    mean_rank.ranks_sum = 10
    mean_rank.num_samples = 5

    mean_rank.reset()

    assert mean_rank.ranks_sum == 0
    assert mean_rank.num_samples == 0


def test_mean_rank_update(mean_rank: MeanRank) -> None:
    scores = torch.tensor([[0.1, 0.5, 0.4], [0.6, 0.2, 0.2], [0.7, 0.1, 0.2]])
    targets = torch.tensor([1, 0, 2])

    mean_rank.update(scores=scores, targets=targets)

    assert mean_rank.ranks_sum == 1 + 1 + 2  # Total rank sum is 4
    assert mean_rank.num_samples == 3  # Total samples is 3


def test_mean_rank_compute(mean_rank: MeanRank) -> None:
    mean_rank.ranks_sum = 9
    mean_rank.num_samples = 3

    mean_rank_value = mean_rank.compute()

    expected_mean_rank = 9 / 3  # The mean rank is 3.0
    assert torch.isclose(mean_rank_value[0], torch.tensor(expected_mean_rank))


def test_mean_rank_compute_no_samples(mean_rank: MeanRank) -> None:
    with pytest.raises(NoSamplesError):
        mean_rank.compute()
