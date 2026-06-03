from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from kge.initialization import XavierConfig, XavierNormal, XavierUniform


class TestXavierConfig:
    """Test XavierConfig validation and creation."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = XavierConfig()
        assert config.gain == 1.0
        assert config.skip_bias_init is False
        assert config.bias_value == 0.0
        assert config.embedding_padding_value == 0.0

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = XavierConfig(
            gain=1.5, skip_bias_init=True, bias_value=0.1, embedding_padding_value=-1.0
        )
        assert config.gain == 1.5
        assert config.skip_bias_init is True
        assert config.bias_value == 0.1
        assert config.embedding_padding_value == -1.0

    def test_invalid_gain(self) -> None:
        """Test that invalid gain values raise validation errors."""
        with pytest.raises(ValueError):
            XavierConfig(gain=0.0)

        with pytest.raises(ValueError):
            XavierConfig(gain=-1.0)


class TestBaseXavier:
    """Test _BaseXavier shared functionality."""

    @pytest.fixture
    def xavier_uniform(self) -> XavierUniform:
        """Create XavierUniform instance for testing."""
        return XavierUniform(gain=1.0, skip_bias_init=False, bias_value=0.0)

    def test_get_fan_in_fan_out_linear(self, xavier_uniform: XavierUniform) -> None:
        """Test fan calculation for linear layers."""
        weight: torch.Tensor = torch.randn(64, 128)
        fan_in, fan_out = xavier_uniform._get_fan_in_fan_out(weight)
        assert fan_in == 128
        assert fan_out == 64

    def test_get_fan_in_fan_out_conv2d(self, xavier_uniform: XavierUniform) -> None:
        """Test fan calculation for 2D convolutional layers."""
        weight: torch.Tensor = torch.randn(32, 16, 3, 3)
        fan_in, fan_out = xavier_uniform._get_fan_in_fan_out(weight)
        assert fan_in == 16 * 3 * 3
        assert fan_out == 32 * 3 * 3

    def test_get_fan_in_fan_out_conv1d(self, xavier_uniform: XavierUniform) -> None:
        """Test fan calculation for 1D convolutional layers."""
        weight: torch.Tensor = torch.randn(64, 32, 5)
        fan_in, fan_out = xavier_uniform._get_fan_in_fan_out(weight)
        assert fan_in == 32 * 5
        assert fan_out == 64 * 5

    def test_get_fan_in_fan_out_invalid_dimensions(self, xavier_uniform: XavierUniform) -> None:
        """Test error handling for tensors with < 2 dimensions."""
        weight: torch.Tensor = torch.randn(10)
        with pytest.raises(ValueError, match="at least 2 dimensions"):
            xavier_uniform._get_fan_in_fan_out(weight)

    def test_init_bias_with_config_normal(self, xavier_uniform: XavierUniform) -> None:
        """Test bias initialization when not skipped."""
        bias: torch.Tensor = torch.randn(10)
        xavier_uniform._init_bias_with_config(bias)
        assert torch.allclose(bias, torch.zeros(10))

    def test_init_bias_with_config_skipped(self) -> None:
        """Test bias initialization when skipped."""
        xavier_uniform: XavierUniform = XavierUniform(skip_bias_init=True)
        bias: torch.Tensor = torch.randn(10)
        original_bias: torch.Tensor = bias.clone()
        xavier_uniform._init_bias_with_config(bias)
        assert torch.allclose(bias, original_bias)

    def test_init_bias_with_config_custom_value(self) -> None:
        """Test bias initialization with custom value."""
        xavier_uniform: XavierUniform = XavierUniform(bias_value=0.5)
        bias: torch.Tensor = torch.randn(10)
        xavier_uniform._init_bias_with_config(bias)
        assert torch.allclose(bias, torch.full((10,), 0.5))

    def test_init_bias_with_config_none(self, xavier_uniform: XavierUniform) -> None:
        """Test bias initialization when bias is None."""
        xavier_uniform._init_bias_with_config(None)


class TestXavierUniform:
    """Test XavierUniform initialization strategy."""

    @pytest.fixture
    def xavier_uniform(self) -> XavierUniform:
        return XavierUniform(gain=1.0)

    def test_initialize_weight_bounds(self, xavier_uniform: XavierUniform) -> None:
        """Test that weights are initialized within expected bounds."""
        weight: torch.Tensor = torch.randn(64, 128)
        fan_in: int = 128
        fan_out: int = 64
        expected_bound: float = math.sqrt(6.0 / (fan_in + fan_out))

        xavier_uniform._initialize_weight(weight)

        assert weight.min() >= -expected_bound
        assert weight.max() <= expected_bound

    def test_initialize_weight_with_gain(self) -> None:
        """Test weight initialization with custom gain."""
        xavier_uniform: XavierUniform = XavierUniform(gain=2.0)
        weight: torch.Tensor = torch.randn(32, 32)
        fan_in: int = 32
        fan_out: int = 32
        expected_bound: float = math.sqrt(6.0 * 4.0 / (fan_in + fan_out))  # gain^2 = 4.0

        xavier_uniform._initialize_weight(weight)

        assert weight.min() >= -expected_bound
        assert weight.max() <= expected_bound

    def test_init_linear(self, xavier_uniform: XavierUniform) -> None:
        """Test linear layer initialization."""
        linear: nn.Linear = nn.Linear(128, 64)
        original_weight: torch.Tensor = linear.weight.clone()

        xavier_uniform.init_linear(linear)

        assert not torch.allclose(linear.weight, original_weight)
        assert torch.allclose(linear.bias, torch.zeros_like(linear.bias))

    def test_init_conv2d(self, xavier_uniform: XavierUniform) -> None:
        """Test 2D convolutional layer initialization."""
        conv: nn.Conv2d = nn.Conv2d(16, 32, kernel_size=3)
        original_weight: torch.Tensor = conv.weight.clone()

        xavier_uniform.init_conv(conv)

        assert not torch.allclose(conv.weight, original_weight)

        assert conv.bias is not None
        assert torch.allclose(conv.bias, torch.zeros_like(conv.bias))

    def test_init_embedding(self, xavier_uniform: XavierUniform) -> None:
        """Test embedding layer initialization."""
        embedding: nn.Embedding = nn.Embedding(1000, 128)
        original_weight: torch.Tensor = embedding.weight.clone()

        xavier_uniform.init_embedding(embedding)

        assert not torch.allclose(embedding.weight, original_weight)

    def test_init_embedding_with_padding(self) -> None:
        """Test embedding initialization with padding token."""
        xavier_uniform: XavierUniform = XavierUniform(embedding_padding_value=-1.0)
        embedding: nn.Embedding = nn.Embedding(1000, 128, padding_idx=0)

        xavier_uniform.init_embedding(embedding)

        assert torch.allclose(embedding.weight[0], torch.full((128,), -1.0))

    def test_from_config(self) -> None:
        """Test creation from configuration."""
        config: XavierConfig = XavierConfig(gain=1.5, skip_bias_init=True)
        xavier_uniform: XavierUniform = XavierUniform.from_config(config)

        assert xavier_uniform.gain == 1.5
        assert xavier_uniform.skip_bias_init is True


class TestXavierNormal:
    """Test XavierNormal initialization strategy."""

    @pytest.fixture
    def xavier_normal(self) -> XavierNormal:
        return XavierNormal(gain=1.0)

    def test_initialize_weight_distribution(self, xavier_normal: XavierNormal) -> None:
        """Test that weights follow expected normal distribution."""
        weight: torch.Tensor = torch.randn(1000, 1000)
        fan_in: int = 1000
        fan_out: int = 1000
        expected_std: float = math.sqrt(2.0 / (fan_in + fan_out))

        xavier_normal._initialize_weight(weight)

        # Check mean is close to 0
        assert abs(weight.mean().item()) < 0.01
        # Check std is close to expected (with some tolerance)
        assert abs(weight.std().item() - expected_std) < 0.01

    def test_initialize_weight_with_gain(self) -> None:
        """Test weight initialization with custom gain."""
        xavier_normal: XavierNormal = XavierNormal(gain=2.0)
        weight: torch.Tensor = torch.randn(1000, 1000)
        fan_in: int = 1000
        fan_out: int = 1000
        expected_std: float = math.sqrt(2.0 * 4.0 / (fan_in + fan_out))  # gain^2 = 4.0

        xavier_normal._initialize_weight(weight)

        assert abs(weight.std().item() - expected_std) < 0.01

    def test_from_config(self) -> None:
        """Test creation from configuration."""
        config: XavierConfig = XavierConfig(gain=0.5, bias_value=0.1)
        xavier_normal: XavierNormal = XavierNormal.from_config(config)

        assert xavier_normal.gain == 0.5
        assert xavier_normal.bias_value == 0.1


class TestIntegration:
    """Integration tests for full model initialization."""

    def test_full_model_initialization(self) -> None:
        """Test initialization on a complete model."""
        model: nn.Sequential = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Conv2d(1, 32, 3),
            nn.BatchNorm2d(32),  # Should be skipped
            nn.Embedding(1000, 128),
            nn.Linear(256, 10),
        )

        xavier_uniform: XavierUniform = XavierUniform(gain=1.0)

        # Store original weights
        original_weights: dict[str, torch.Tensor] = {}
        for name, module in model.named_modules():
            if hasattr(module, "weight"):
                original_weights[name] = module.weight.clone()

        # Apply initialization
        xavier_uniform(model)

        # Check that weights were changed (except BatchNorm)
        for name, module in model.named_modules():
            if hasattr(module, "weight") and not isinstance(module, nn.BatchNorm2d):
                assert not torch.allclose(module.weight, original_weights[name])

    @pytest.mark.parametrize("gain", [0.5, 1.0, 1.5, 2.0])
    def test_different_gains(self, gain: float) -> None:
        """Test initialization with different gain values."""
        linear: nn.Linear = nn.Linear(100, 50)
        xavier_uniform: XavierUniform = XavierUniform(gain=gain)

        xavier_uniform.init_linear(linear)

        fan_in: int = 100
        fan_out: int = 50
        expected_bound: float = math.sqrt(6.0 * gain * gain / (fan_in + fan_out))
        assert linear.weight.min() >= -expected_bound
        assert linear.weight.max() <= expected_bound
