import pytest
import torch

from hakken_ml_toolkit.losses import BCEWithLogitsLoss, BCEWithLogitsLossConfig
from hakken_ml_toolkit.losses.base.clf_loss import ClfLossConfig, ClfLossI
from hakken_ml_toolkit.losses.common.domain import is_float_tensor_with_dim


@pytest.fixture
def bce_loss():
    config = BCEWithLogitsLossConfig()
    return BCEWithLogitsLoss(config)


def test_binary_classification(bce_loss: BCEWithLogitsLoss):
    logits = torch.tensor([[0.0], [0.0], [0.0]])
    target = torch.tensor([[1], [0], [1]])

    loss = bce_loss._compute(logits, target)

    assert is_float_tensor_with_dim(loss, dim=1)
    assert loss.shape == torch.Size([3])
    loss_true = torch.tensor(
        [
            0.6931,
        ]
        * 3
    )
    assert torch.allclose(
        loss,
        loss_true,
        atol=1e-3,
    ), f"loss: {loss} {loss_true}"


def test_multilabel_classification(bce_loss: BCEWithLogitsLoss):
    logits = torch.tensor([[0.5, -0.5, 1.0], [-1.0, 0.2, 0.7]])
    target = torch.tensor([[1, 0, 1], [0, 1, 1]])

    loss = bce_loss._compute(logits, target)

    assert is_float_tensor_with_dim(loss, dim=1)
    assert loss.shape == torch.Size([2])


def test_with_sample_weights(bce_loss: BCEWithLogitsLoss):
    logits = torch.tensor([[0.5], [-0.5], [1.0]])
    target = torch.tensor([[1], [0], [1]])
    weights = torch.tensor([0.5, 1.0, 2.0]).unsqueeze(1)

    loss = bce_loss._compute(logits, target, samples_weight=weights)

    assert is_float_tensor_with_dim(loss, dim=1)

    assert loss.shape == torch.Size([3])


def test_zero_loss(bce_loss: BCEWithLogitsLoss):
    logits = torch.tensor([[100.0], [-100.0]])
    target = torch.tensor([[1], [0]])

    loss = bce_loss._compute(logits, target)

    assert is_float_tensor_with_dim(loss, dim=1)

    assert loss.shape == torch.Size([2])
    assert torch.all(loss < 1e-6)


def test_config():
    config = BCEWithLogitsLossConfig()
    assert isinstance(config, ClfLossConfig)


def test_inheritance():
    config = BCEWithLogitsLossConfig()
    loss = BCEWithLogitsLoss(config)
    assert isinstance(loss, ClfLossI)
