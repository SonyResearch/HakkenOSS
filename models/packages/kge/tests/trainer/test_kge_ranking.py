from typing import cast
from unittest.mock import MagicMock, patch

import pytest
import torch
from hakken_ml_toolkit.losses import RankingLossI
from hakken_ml_toolkit.tracker import TrackerI
from torch import Tensor

from kge.evaluator.base import KGEEvaluator
from kge.models.base import KGEI
from kge.negative_sampler.base import NegativeSamplerI
from kge.optim.factory import LRSchedulerInfo, OptimizerInfo
from kge.trainer.lightning.ranking import (
    TRAINING_LOSS_KEY,
    VALIDATION_LOSS_KEY,
    KGERankingLightning,
)


@pytest.fixture
def mock_model() -> MagicMock:
    model = MagicMock(spec=KGEI)
    model.return_value = MagicMock(scores=torch.tensor([[0.5], [0.3]]))
    return model


optimizer_info = OptimizerInfo(class_name="torch.optim.Adam", kwargs={"lr": 0.00})

lr_sched_info = LRSchedulerInfo(
    class_name="torch.optim.lr_scheduler.ReduceLROnPlateau",
    kwargs={"mode": "min", "factor": 0.9, "patience": 5},
)


@pytest.fixture
def mock_negative_sampler() -> MagicMock:
    sampler = MagicMock(spec=NegativeSamplerI)
    # Create a mock tensor that simulates negative samples
    neg_samples = torch.Tensor([[[1, 2, 3], [4, 5, 6]]]).reshape(2, 1, 3)
    sampler.corrupt_batch.return_value = neg_samples

    return sampler


@pytest.fixture
def mock_loss_fn() -> MagicMock:
    loss_fn = MagicMock(spec=RankingLossI)
    loss_fn.compute.return_value = torch.tensor(0.5)

    return loss_fn


@pytest.fixture
def mock_tracker() -> MagicMock:
    tracker = MagicMock(spec=TrackerI)
    tracker.increment_step = MagicMock()
    return tracker


@pytest.fixture
def mock_evaluator() -> MagicMock:
    return MagicMock(spec=KGEEvaluator)


@pytest.fixture
def kge_lightning(
    mock_model: MagicMock,
    mock_negative_sampler: MagicMock,
    mock_loss_fn: MagicMock,
    mock_tracker: MagicMock,
    mock_evaluator: MagicMock,
) -> KGERankingLightning:
    return KGERankingLightning(
        model=cast("KGEI", mock_model),
        negative_sampler=cast("NegativeSamplerI", mock_negative_sampler),
        optimizer_info=optimizer_info,
        lr_sched_info=lr_sched_info,
        loss_fn=cast("RankingLossI", mock_loss_fn),
        tracker=cast("TrackerI", mock_tracker),
        evaluator=cast("KGEEvaluator", mock_evaluator),
    )


def test_initialization(
    kge_lightning: KGERankingLightning,
    mock_model: MagicMock,
    mock_negative_sampler: MagicMock,
    mock_loss_fn: MagicMock,
    mock_tracker: MagicMock,
    mock_evaluator: MagicMock,
) -> None:
    assert kge_lightning.model == mock_model
    assert kge_lightning.negative_sampler == mock_negative_sampler
    assert kge_lightning.loss_fn == mock_loss_fn
    assert kge_lightning.tracker == mock_tracker
    assert kge_lightning.evaluator == mock_evaluator
    assert kge_lightning.sro_remove is None


def test_initialization_with_remove_triples(
    mock_model: MagicMock,
    mock_negative_sampler: MagicMock,
    mock_loss_fn: MagicMock,
    mock_tracker: MagicMock,
    mock_evaluator: MagicMock,
) -> None:
    with patch("hakken_ml_toolkit.ml_utils.extras.PyTorchUtils.load") as mock_load:
        mock_load.return_value = MagicMock()
        kge_lightning = KGERankingLightning(
            model=cast("KGEI", mock_model),
            negative_sampler=cast("NegativeSamplerI", mock_negative_sampler),
            optimizer_info=optimizer_info,
            lr_sched_info=lr_sched_info,
            loss_fn=cast("RankingLossI", mock_loss_fn),
            tracker=cast("TrackerI", mock_tracker),
            evaluator=cast("KGEEvaluator", mock_evaluator),
            remove_triples_path="dummy/path",
        )
        assert kge_lightning.sro_remove is not None
        mock_load.assert_called_once_with("dummy/path")


def test_training_step(kge_lightning: KGERankingLightning) -> None:
    batch: list[Tensor] = [torch.Tensor([[1, 2, 3], [4, 5, 6]])]

    mock_opt = MagicMock()
    kge_lightning.optimizers = lambda: mock_opt  # type: ignore

    with patch.object(kge_lightning, "manual_backward", MagicMock()) as mock_backward:
        result = kge_lightning.training_step(batch, 0)

        assert TRAINING_LOSS_KEY in result
        assert isinstance(result[TRAINING_LOSS_KEY], torch.Tensor)
        assert mock_opt.zero_grad.called
        assert mock_opt.step.called
        assert mock_backward.called


def test_validation_step(kge_lightning: KGERankingLightning) -> None:
    batch: list[Tensor] = [torch.tensor([[1, 2, 3], [4, 5, 6]])]

    result = kge_lightning.validation_step(batch, 0)

    assert VALIDATION_LOSS_KEY in result
    assert isinstance(result[VALIDATION_LOSS_KEY], torch.Tensor)


def test_training_step_with_remove_triples(
    mock_model: MagicMock,
    mock_negative_sampler: MagicMock,
    mock_loss_fn: MagicMock,
    mock_tracker: MagicMock,
    mock_evaluator: MagicMock,
) -> None:
    mock_remove_triples = torch.Tensor([[1, 2, 3]])
    kge_lightning = KGERankingLightning(
        model=cast("KGEI", mock_model),
        negative_sampler=cast("NegativeSamplerI", mock_negative_sampler),
        optimizer_info=optimizer_info,
        lr_sched_info=lr_sched_info,
        loss_fn=cast("RankingLossI", mock_loss_fn),
        tracker=cast("TrackerI", mock_tracker),
        evaluator=cast("KGEEvaluator", mock_evaluator),
    )
    kge_lightning.sro_remove = mock_remove_triples

    mock_opt = MagicMock()
    kge_lightning.optimizers = lambda: mock_opt  # type: ignore

    with patch.object(kge_lightning, "manual_backward", MagicMock()) as mock_backward:
        batch: list[Tensor] = [torch.tensor([[1, 2, 3], [4, 5, 6]])]

        result = kge_lightning.training_step(batch, 0)

        assert TRAINING_LOSS_KEY in result
        assert isinstance(result[TRAINING_LOSS_KEY], torch.Tensor)
        assert mock_opt.zero_grad.called
        assert mock_opt.step.called
        assert mock_backward.called


def test_multiple_optimizers_error(kge_lightning: KGERankingLightning) -> None:
    kge_lightning.optimizers = lambda: [MagicMock(), MagicMock()]  # type: ignore

    batch: list[Tensor] = [torch.tensor([[1, 2, 3], [4, 5, 6]])]

    with pytest.raises(Exception, match="Only one optimizer is allowed"):
        kge_lightning.training_step(batch, 0)
