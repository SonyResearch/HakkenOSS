import math
from collections.abc import Iterator
from pathlib import Path

import pytest

try:
    import torch
except ImportError:
    pytest.skip("PyTorch is not installed", allow_module_level=True)


from hakken_ml_toolkit.ml_utils.extras.domain import TensorND
from hakken_ml_toolkit.ml_utils.extras.scalers import (
    MinMaxScaler,
    MinMaxScalerConfig,
    ScalerI,
)
from hakken_ml_toolkit.ml_utils.extras.scalers.core.values.exceptions import (
    ScalerNotFittedError,
    ZeroRangeError,
)


@pytest.fixture
def sample_data() -> TensorND:
    return torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])


@pytest.fixture
def sample_iterator() -> Iterator[TensorND]:
    data = [
        torch.tensor([[1.0, 2.0, 3.0]]),
        torch.tensor([[4.0, 5.0, 6.0]]),
        torch.tensor([[7.0, 8.0, 9.0]]),
    ]
    return iter(data)


def test_minmax_scaler_init() -> None:
    config = MinMaxScalerConfig(feature_range=(0, 1))
    scaler = MinMaxScaler(config)
    assert isinstance(scaler, ScalerI)
    assert scaler.feature_range == (0, 1)


def test_minmax_scaler_fit(sample_data: TensorND) -> None:
    scaler = MinMaxScaler(MinMaxScalerConfig())
    scaler.fit(sample_data)
    assert torch.allclose(scaler.data_min_, torch.tensor([1.0, 2.0, 3.0]))  # type: ignore
    assert torch.allclose(scaler.data_max_, torch.tensor([7.0, 8.0, 9.0]))  # type: ignore
    assert torch.allclose(scaler.data_range_, torch.tensor([6.0, 6.0, 6.0]))  # type: ignore


def test_minmax_scaler_fit_from_iterator(sample_iterator: Iterator[TensorND]) -> None:
    scaler = MinMaxScaler(MinMaxScalerConfig())
    scaler.fit_from_iterator(sample_iterator)
    assert torch.allclose(scaler.data_min_, torch.tensor([1.0, 2.0, 3.0]))  # type: ignore
    assert torch.allclose(scaler.data_max_, torch.tensor([7.0, 8.0, 9.0]))  # type: ignore
    assert torch.allclose(scaler.data_range_, torch.tensor([6.0, 6.0, 6.0]))  # type: ignore


def test_minmax_scaler_transform(sample_data: TensorND) -> None:
    scaler = MinMaxScaler(MinMaxScalerConfig())
    scaler.fit(sample_data)
    transformed = scaler.transform(sample_data)
    expected = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]])

    assert torch.allclose(transformed, expected)


def test_minmax_scaler_inverse_transform(sample_data: TensorND) -> None:
    scaler = MinMaxScaler(MinMaxScalerConfig())
    scaler.fit(sample_data)
    transformed = scaler.transform(sample_data)
    inverse_transformed = scaler.inverse_transform(transformed)
    assert torch.allclose(inverse_transformed, sample_data)


def test_minmax_scaler_feature_range() -> None:
    config = MinMaxScalerConfig(feature_range=(-1, 1))
    scaler = MinMaxScaler(config)
    data = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    scaler.fit(data)
    transformed = scaler.transform(data)
    expected = torch.tensor([[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    assert torch.allclose(transformed, expected)


def test_minmax_scaler_error_before_fit() -> None:
    scaler = MinMaxScaler(MinMaxScalerConfig())
    data = torch.tensor([[1.0, 2.0, 3.0]])
    with pytest.raises(ScalerNotFittedError):
        scaler.transform(data)


def test_minmax_scaler_save_load(tmp_path: Path, sample_data: TensorND) -> None:
    scaler = MinMaxScaler(MinMaxScalerConfig())
    scaler.fit(sample_data)

    save_path = tmp_path / "scaler.json"
    scaler.save(save_path)

    loaded_scaler = MinMaxScaler.load(str(save_path))
    assert isinstance(loaded_scaler, MinMaxScaler)
    assert torch.allclose(loaded_scaler.data_min_, scaler.data_min_)  # type: ignore
    assert torch.allclose(loaded_scaler.data_max_, scaler.data_max_)  # type: ignore
    assert torch.allclose(loaded_scaler.data_range_, scaler.data_range_)  # type: ignore


@pytest.mark.parametrize("feature_range", [(-1, 1), (0, 10), (-10, 10)])
def test_minmax_scaler_different_ranges(
    feature_range: tuple[float, float], sample_data: TensorND
) -> None:
    config = MinMaxScalerConfig(feature_range=feature_range)
    scaler = MinMaxScaler(config)
    scaler.fit(sample_data)
    transformed = scaler.transform(sample_data)
    min_data = torch.min(transformed).item()
    max_data = torch.max(transformed).item()
    assert math.isclose(min_data, feature_range[0], rel_tol=1e-5)
    assert math.isclose(max_data, feature_range[1], rel_tol=1e-5)


def test_minmax_scaler_with_constant_feature(sample_data: TensorND) -> None:
    constant_data = torch.cat([sample_data, torch.ones((3, 1))], dim=1)
    scaler = MinMaxScaler(MinMaxScalerConfig())

    with pytest.raises(ZeroRangeError):
        scaler.fit(constant_data)

    # Test with non-constant data to ensure it still works
    scaler = MinMaxScaler(MinMaxScalerConfig())
    scaler.fit(sample_data)  # This should not raise an error
    transformed = scaler.transform(sample_data)
    assert transformed.shape == sample_data.shape
