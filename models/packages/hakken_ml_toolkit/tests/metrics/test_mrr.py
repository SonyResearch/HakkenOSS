import pytest

from hakken_ml_toolkit.metrics.core.create_tensors import TensorCreator
from hakken_ml_toolkit.metrics.core.exceptions import NoSamplesError
from hakken_ml_toolkit.metrics.impl.mrr import MeanReciprocalRank, MRRConfig


@pytest.fixture
def mrr_metric() -> MeanReciprocalRank:
    config = MRRConfig()
    return MeanReciprocalRank(config=config)


def test_single_sample_correct_first(mrr_metric: MeanReciprocalRank) -> None:
    scores = TensorCreator.float_tensor([[0.9, 0.8, 0.7, 0.6, 0.5]])

    targets = TensorCreator.long_tensor([0])
    mrr_metric.reset()
    mrr_metric.update(scores=scores, targets=targets)
    result = mrr_metric.compute()
    expected_mrr = 1.0  # Reciprocal rank is 1 / 1 = 1.0
    assert result.item() == pytest.approx(expected_mrr, abs=1e-5)


def test_single_sample_correct_last(mrr_metric: MeanReciprocalRank) -> None:
    scores = TensorCreator.float_tensor([[0.5, 0.6, 0.7, 0.8, 0.9]])

    targets = TensorCreator.long_tensor([0])
    mrr_metric.reset()
    mrr_metric.update(scores=scores, targets=targets)
    result = mrr_metric.compute()
    expected_mrr = 1.0 / 5.0  # Reciprocal rank is 1 / 5 = 0.2
    assert result.item() == pytest.approx(expected_mrr, abs=1e-5)


def test_multiple_samples(mrr_metric: MeanReciprocalRank) -> None:
    scores = TensorCreator.float_tensor(
        [
            [0.2, 0.9, 0.3, 0.4, 0.5],
            [0.1, 0.2, 0.3, 0.9, 0.5],
            [0.9, 0.1, 0.2, 0.3, 0.4],
            [0.5, 0.4, 0.9, 0.2, 0.1],
        ],
    )
    targets = TensorCreator.long_tensor([1, 3, 0, 0])  # Correct items
    mrr_metric.reset()
    mrr_metric.update(scores=scores, targets=targets)
    result = mrr_metric.compute()

    expected_mrr = (1.0 + 1.0 + 1.0 + 0.5) / 4.0  # MRR = 3.5 / 4 = 0.875
    assert result.item() == pytest.approx(expected_mrr, abs=1e-5)


def test_no_samples(mrr_metric: MeanReciprocalRank) -> None:
    mrr_metric.reset()
    with pytest.raises(NoSamplesError):
        mrr_metric.compute()


def test_ties_in_scores(mrr_metric: MeanReciprocalRank) -> None:
    scores = TensorCreator.float_tensor(
        [
            [0.9, 0.9, 0.8, 0.7, 0.6],
            [0.5, 0.5, 0.5, 0.5, 0.5],
        ]
    )

    targets = TensorCreator.long_tensor([0, 3])
    mrr_metric.reset()
    mrr_metric.update(scores=scores, targets=targets)
    result = mrr_metric.compute()

    expected_mrr = (1.0 + 1.0) / 2.0  # MRR = 1.25 / 2 = 0.625
    assert result.item() == pytest.approx(expected_mrr, abs=1e-5)
