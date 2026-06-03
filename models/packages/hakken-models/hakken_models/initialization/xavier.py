from __future__ import annotations

import math

import torch
from pydantic import Field
from torch import nn

from .base import BaseInitStrategy, InitStrategyConfig


class XavierConfig(InitStrategyConfig):
    """Base configuration for Xavier initialization strategies."""

    gain: float = Field(
        default=1.0,
        gt=0.0,
        description="Scaling factor applied to the computed variance",
    )

    skip_bias_init: bool = Field(default=False, description="Whether to skip bias initialization")

    bias_value: float = Field(default=0.0, description="Value to initialize biases to")

    embedding_padding_value: float = Field(
        default=0.0, description="Value for padding tokens in embeddings"
    )


class _BaseXavier(BaseInitStrategy[XavierConfig]):
    """Base class for Xavier initialization strategies with shared logic."""

    def __init__(
        self,
        gain: float = 1.0,
        skip_bias_init: bool = False,
        bias_value: float = 0.0,
        embedding_padding_value: float = 0.0,
    ):
        """Initialize Xavier strategy base."""
        self.gain = gain
        self.skip_bias_init = skip_bias_init
        self.bias_value = bias_value
        self.embedding_padding_value = embedding_padding_value

    def _get_fan_in_fan_out(self, tensor: torch.Tensor) -> tuple[int, int]:
        """
        Calculate fan_in and fan_out for a tensor.

        Args:
            tensor: Weight tensor to analyze

        Returns:
            Tuple of (fan_in, fan_out)
        """
        dimensions = tensor.dim()

        if dimensions < 2:
            msg = f"Tensor must have at least 2 dimensions, got {dimensions}"
            raise ValueError(msg)

        # For different tensor shapes:
        if dimensions == 2:  # Linear layer: (out_features, in_features)
            fan_in = tensor.size(1)
            fan_out = tensor.size(0)
        else:  # Convolutional layers: (out_channels, in_channels, kernel_sizes...)
            num_input_fmaps = tensor.size(1)
            num_output_fmaps = tensor.size(0)
            receptive_field_size = 1

            # Calculate receptive field size (product of kernel dimensions)
            if tensor.dim() > 2:
                for s in tensor.shape[2:]:
                    receptive_field_size *= s

            fan_in = num_input_fmaps * receptive_field_size
            fan_out = num_output_fmaps * receptive_field_size

        return fan_in, fan_out

    def _init_bias_with_config(self, bias: torch.Tensor | None) -> None:
        """Initialize bias according to configuration."""
        if bias is not None and not self.skip_bias_init:
            nn.init.constant_(bias, self.bias_value)

    def init_conv(self, module: nn.Conv1d | nn.Conv2d | nn.Conv3d) -> None:
        """Initialize convolutional layer with Xavier initialization."""
        if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
            self._initialize_weight(module.weight)
        self._init_bias_with_config(module.bias)

    def init_linear(self, module: nn.Linear) -> None:
        """Initialize linear layer with Xavier initialization."""
        if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
            self._initialize_weight(module.weight)
        self._init_bias_with_config(module.bias)

    def init_embedding(self, module: nn.Embedding) -> None:
        """Initialize embedding layer with Xavier initialization."""
        if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
            self._initialize_weight(module.weight)

        # Handle padding token
        if hasattr(module, "padding_idx") and module.padding_idx is not None:
            with torch.no_grad():
                module.weight[module.padding_idx].fill_(self.embedding_padding_value)

    def _initialize_weight(self, weight: torch.Tensor) -> None:
        """Initialize weight tensor - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _initialize_weight")


class XavierUniform(_BaseXavier):
    """
    Xavier Uniform initialization strategy.

    Implements Xavier uniform initialization that preserves activation
    and gradient variance by sampling from:
    U(±√(6*gain²/(fan_in + fan_out)))
    """

    def _initialize_weight(self, weight: torch.Tensor) -> None:
        """Initialize weight tensor using Xavier uniform distribution."""
        fan_in, fan_out = self._get_fan_in_fan_out(weight)

        # Calculate the bound: √(6*gain²/(fan_in + fan_out))
        bound = math.sqrt(6.0 * self.gain * self.gain / (fan_in + fan_out))

        with torch.no_grad():
            weight.uniform_(-bound, bound)


class XavierNormal(_BaseXavier):
    """
    Xavier Normal initialization strategy.

    Implements Xavier normal initialization that preserves activation
    and gradient variance by sampling from:
    N(0, √(2*gain²/(fan_in + fan_out)))
    """

    def _initialize_weight(self, weight: torch.Tensor) -> None:
        """Initialize weight tensor using Xavier normal distribution."""
        fan_in, fan_out = self._get_fan_in_fan_out(weight)

        # Calculate the standard deviation: √(2*gain²/(fan_in + fan_out))
        std = math.sqrt(2.0 * self.gain * self.gain / (fan_in + fan_out))

        with torch.no_grad():
            weight.normal_(0.0, std)
