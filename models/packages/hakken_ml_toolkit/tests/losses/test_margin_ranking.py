import pytest
import torch

from hakken_ml_toolkit.losses import MarginRankingLoss, MarginRankingLossConfig
from hakken_ml_toolkit.losses.common.domain import is_float_tensor_with_dim
from hakken_ml_toolkit.losses.common.exceptions import ShapeMismatchError


@pytest.fixture
def margin_ranking_loss():
    config = MarginRankingLossConfig(margin=1.0, reduce="none")
    return MarginRankingLoss(config)


def test_margin_ranking_loss_compute():
    config = MarginRankingLossConfig(margin=1.0, reduce="none")
    loss_fn = MarginRankingLoss(config)

    positive_scores = torch.tensor([[0.5], [0.7], [0.9]])
    negative_scores = torch.tensor([[0.3], [0.8], [0.6]])

    expected_losses = torch.tensor([0.8, 1.1, 0.7])

    result = loss_fn.compute(positive_scores, negative_scores)

    assert is_float_tensor_with_dim(result, dim=1)
    assert torch.allclose(result, expected_losses)


def test_margin_ranking_loss_reduction():
    loss_fn_sum = MarginRankingLoss(MarginRankingLossConfig(margin=1.0, reduce="sum"))
    loss_fn_mean = MarginRankingLoss(MarginRankingLossConfig(margin=1.0, reduce="mean"))

    losses = torch.tensor([0.8, 1.1, 0.7])

    sum_result = loss_fn_sum.reduce(losses)
    mean_result = loss_fn_mean.reduce(losses)

    assert torch.allclose(sum_result, torch.tensor([2.6]))
    assert torch.allclose(mean_result, torch.tensor([2.6 / 3]))


def test_margin_ranking_loss_zero_margin():
    loss_fn = MarginRankingLoss(MarginRankingLossConfig(margin=0.0, reduce="none"))

    positive_scores = torch.tensor([[0.5], [0.7], [0.9]])
    negative_scores = torch.tensor([[0.3], [0.8], [0.6]])

    expected_losses = torch.tensor([0.0, 0.1, 0.0])

    result = loss_fn._compute(positive_scores, negative_scores)

    assert torch.allclose(result, expected_losses)


def test_margin_ranking_loss_large_margin():
    loss_fn = MarginRankingLoss(MarginRankingLossConfig(margin=2.0, reduce="none"))

    positive_scores = torch.tensor([[0.5], [0.7], [0.9]])
    negative_scores = torch.tensor([[0.3], [0.8], [0.6]])

    expected_losses = torch.tensor([1.8, 2.1, 1.7])

    result = loss_fn._compute(positive_scores, negative_scores)

    assert torch.allclose(result, expected_losses)


def test_margin_ranking_loss_invalid_input():
    loss_fn = MarginRankingLoss(MarginRankingLossConfig(margin=1.0, reduce="none"))

    with pytest.raises(ShapeMismatchError):
        invalid_scores = torch.tensor([[0.5, 0.6], [0.7, 0.8]])
        loss_fn.compute(invalid_scores, invalid_scores)


def test_margin_ranking_loss_different_batch_sizes():
    loss_fn = MarginRankingLoss(MarginRankingLossConfig(margin=1.0, reduce="none"))

    positive_scores = torch.tensor([[0.5], [0.7], [0.9]])
    negative_scores = torch.tensor([[0.3], [0.8]])

    with pytest.raises(ShapeMismatchError):
        loss_fn.compute(positive_scores, negative_scores)
