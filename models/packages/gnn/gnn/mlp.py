from __future__ import annotations

from typing import Annotated, cast

import torch
from hakken_ml_toolkit.ml_utils.constants import ActivationType
from hakken_ml_toolkit.ml_utils.extras import PyTorchUtils
from pydantic import BaseModel, Field
from torch import nn

from gnn.common.exceptions import WrongDimensionsError


class LayerConfig(BaseModel):
    input_dim: Annotated[int, Field(gt=0, description="Dimension of the input features.")]
    output_dim: Annotated[int, Field(gt=0, description="Dimension of the output features.")]
    activation: nn.Module = Field(..., description="PyTorch activation module (e.g., nn.ReLU()).")

    class Config:
        arbitrary_types_allowed = True


class MLPConfig(BaseModel):
    input_dim: int
    output_dim: int
    num_layers: int
    activation: ActivationType = ActivationType.RELU
    hidden_dim: int | None = None
    batch_norm: bool = False
    layer_norm: bool = False
    use_activation_output: bool = False
    dropout: float = 0.0
    drop_last: bool = False
    device: str | torch.device = "cpu"

    class Config:
        arbitrary_types_allowed = True


class MLP(nn.Module):
    """
    Multi-Layer Perceptron implementation that supports variable depth, different activation
    functions, normalization layers, and dropout.
    """

    def __init__(self, config: MLPConfig):
        super().__init__()

        self.config = config

        layers: list[nn.Module] = []
        for n in range(config.num_layers):
            layer_config = self._get_layer_config(
                layer_idx=n,
                total_layers=config.num_layers,
                input_dim=config.input_dim,
                hidden_dim=config.hidden_dim,
                output_dim=config.output_dim,
                activation=config.activation,
                use_activation_output=config.use_activation_output,
                device=config.device,
            )

            blocks: list[nn.Module] = []

            blocks.append(
                nn.Linear(
                    layer_config.input_dim,
                    layer_config.output_dim,
                    device=self.config.device,
                )
            )
            if config.batch_norm:
                blocks.append(nn.BatchNorm1d(layer_config.output_dim, device=config.device))
            elif config.layer_norm:
                blocks.append(nn.LayerNorm(layer_config.output_dim, device=config.device))
            blocks.append(layer_config.activation)

            if config.dropout > 0.0 and (n < (config.num_layers - 1) or config.drop_last):
                drop = nn.Dropout(config.dropout).to(config.device)
                blocks.append(drop)
            layers.append(nn.Sequential(*blocks).to(config.device))

        if config.num_layers == 0:
            layers = [nn.Identity().to(config.device)]
            if config.input_dim != config.output_dim:
                msg = f"{config.input_dim}!= {config.output_dim}"
                raise WrongDimensionsError(msg)

        self.fc_layers = nn.ModuleList(layers)

    def _get_layer_config(
        self,
        layer_idx: int,
        total_layers: int,
        input_dim: int,
        hidden_dim: int | None,
        output_dim: int,
        activation: ActivationType,
        use_activation_output: bool,
        device: str | torch.device,
    ) -> LayerConfig:
        """Determines the configuration for a specific layer in the MLP.

        Args:
            layer_idx: Index of the current layer.
            total_layers: Total number of layers in the MLP.
            input_dim: Input dimension of the MLP.
            hidden_dim: Hidden dimension size.
            output_dim: Output dimension of the MLP.
            activation: Type of activation function to use.
            use_activation_output: Whether to apply activation to the output layer.
            device: Device to place the layer on.

        Returns:
            A dictionary with the layer configuration including input_dim,
            output_dim, and activation function.

        Raises:
            ValueError: If total_layers > 1 and hidden_dim is not provided.
        """
        if total_layers > 1 and hidden_dim is None:
            msg = "hidden_dim must be provided if total_layers>1"
            raise ValueError(msg)

        if layer_idx == 0:
            if total_layers == 1:
                return LayerConfig(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    activation=(
                        PyTorchUtils.activation(activation)
                        if use_activation_output
                        else nn.Identity().to(device)
                    ),
                )

            return LayerConfig(
                input_dim=input_dim,
                output_dim=cast("int", hidden_dim),
                activation=PyTorchUtils.activation(activation),
            )

        if layer_idx == (total_layers - 1):
            return LayerConfig(
                input_dim=cast("int", hidden_dim),
                output_dim=output_dim,
                activation=(
                    PyTorchUtils.activation(activation)
                    if use_activation_output
                    else nn.Identity().to(device)
                ),
            )

        return LayerConfig(
            input_dim=cast("int", hidden_dim),
            output_dim=cast("int", hidden_dim),
            activation=PyTorchUtils.activation(activation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after passing through all MLP layers.
        """
        for fc in self.fc_layers:
            x = fc(x)

        return x
