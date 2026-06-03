from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Annotated, Any, Generic, TypeVar

import torch
from hakken_ml_toolkit.ml_utils.constants import ActivationType
from hakken_ml_toolkit.ml_utils.extras import PyTorchUtils
from pydantic import BaseModel, Field, field_validator
from torch import nn

from gnn.common.constants import PoolingType, StageType
from gnn.mlp import MLP, MLPConfig
from gnn.node_wrapper import NodeWrapper
from gnn.pooling import GraphPooling, GraphPoolingConfig

if TYPE_CHECKING:
    import torch_geometric.data as pygd

    from gnn.common.domain import FloatTensor2D


class GNNConfig(BaseModel):
    input_dim: Annotated[int, Field(gt=0, description="Dimension of the input features.")]
    output_dim: Annotated[int, Field(gt=0, description="Dimension of the output features.")]
    hidden_dim: Annotated[int, Field(ge=0, description="Dimension of hidden layers.")] = 32

    activation: ActivationType = Field(
        default=ActivationType.RELU, description="Activation function to use."
    )
    dropout: Annotated[float, Field(ge=0.0, le=1.0, description="Dropout rate (0.0 to 1.0).")] = 0.0
    batch_norm: bool = Field(default=False, description="Whether to use batch normalization.")

    stage_type: StageType = Field(
        default=StageType.SKIPSUM, description="Stage type to use in the GNN."
    )

    num_layers_pre: Annotated[
        int, Field(ge=0, description="Number of pre-MLP layers. Set to 0 to disable.")
    ] = 1
    num_layers_gnn: Annotated[
        int, Field(ge=0, description="Number of GNN layers. Set to 0 to disable.")
    ] = 1
    num_layers_post: Annotated[
        int, Field(ge=0, description="Number of post-MLP layers. Set to 0 to disable.")
    ] = 1

    pooling: list[PoolingType] | None = Field(
        default=None, description="List of pooling strategies to apply."
    )
    device: str | torch.device = Field(
        default="cpu", description="Computation device (e.g., 'cpu' or 'cuda')."
    )

    class Config:
        arbitrary_types_allowed = True

    @field_validator("dropout")
    @classmethod
    def validate_dropout(cls, v):
        if not 0 <= v <= 1:
            msg = "Dropout must be between 0 and 1"
            raise ValueError(msg)
        return v


T = TypeVar("T", bound=GNNConfig)


class GNNI(nn.Module, ABC, Generic[T]):
    """
    Abstract base class for Graph Neural Network implementations.

    This class provides customizable pre-processing, graph convolution,
    and post-processing stages.The architecture consists of three main
    components:

    1. Pre-processing MLP: Transforms node features before graph convolutions
    2. GNN layers: Performs message passing between nodes (implementation dependent
        on subclass)
    3. Post-processing MLP: Transforms node features after graph convolutions
    4. Optional pooling: Aggregates node features to graph-level representations


    Notes
    -----
    Subclasses must implement the _gnn_layer method to define the specific
    graph convolution operation to be used.

    The class is designed to work with PyTorch Geometric data structures.
    """

    def __init__(self, config: T):
        super().__init__()

        self.has_pre_mlp = config.num_layers_pre > 0
        self.has_gnn = config.num_layers_gnn > 0
        self.has_post_mlp = config.num_layers_post > 0

        self.config = config
        self.lin_skipsum: nn.ModuleList | None = None

        if self.has_pre_mlp:
            pre_input_dim = config.input_dim
            pre_output_dim = config.hidden_dim
        else:
            pre_input_dim = config.input_dim
            pre_output_dim = config.input_dim

        pre_config = MLPConfig(
            input_dim=pre_input_dim,
            output_dim=pre_output_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers_pre,
            activation=config.activation,
            batch_norm=config.batch_norm,
            dropout=config.dropout,
            use_activation_output=self.has_gnn or self.has_post_mlp,
            device=config.device,
        )
        self.pre_nn = MLP(pre_config)

        gnn_output_dim = config.hidden_dim if config.num_layers_post > 0 else config.output_dim

        self.gnn = self._build_gnn(
            input_dim=self.pre_nn.config.output_dim,
            hidden_dim=config.hidden_dim,
            output_dim=gnn_output_dim,
            activation=config.activation,
            dropout=config.dropout,
            layers_num=config.num_layers_gnn,
            act_last=config.num_layers_post > 0,
        )

        if config.num_layers_gnn > 0:
            input_dim_post = config.hidden_dim
        else:
            input_dim_post = self.pre_nn.config.output_dim

        output_dim_post = config.hidden_dim if config.pooling is not None else config.output_dim

        post_config = MLPConfig(
            input_dim=input_dim_post,
            output_dim=output_dim_post,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers_post,
            activation=config.activation,
            batch_norm=config.batch_norm,
            dropout=config.dropout,
            drop_last=True,
            device=config.device,
        )

        self.post_nn = MLP(post_config)

        self.pooling: GraphPooling | None = None
        if config.pooling is not None:
            pooling_config = GraphPoolingConfig(
                pool_type=config.pooling,
                in_channels=output_dim_post,
                activation=config.activation,
                out_channels=config.output_dim,
                batch_norm=False,
            )
            self.pooling = GraphPooling(pooling_config)

    def forward(self, batch: pygd.Batch, inplace: bool = False, **kwargs) -> FloatTensor2D:
        """
        Forward pass through the entire GNN architecture.

        Args:
            batch: PyTorch Geometric batch containing graph data
            inplace: Whether to modify the input batch in-place
            **kwargs: Additional arguments to pass to GNN layers

        Returns:
            Node features or graph-level representation
        """
        original_x = None
        if not inplace:
            original_x = batch.x.clone()

        batch = self.pre_forward(batch)

        batch = self.forward_gnn(batch, **kwargs)
        batch = self.post_forward(batch)

        output: FloatTensor2D = batch.x if self.pooling is None else self.pooling(batch)

        if original_x is not None:
            batch.x = original_x

        return output

    def pre_forward(self, batch: pygd.Batch) -> pygd.Batch:
        """
        Forward pass through the pre-processing MLP.

        Args:
            batch: PyTorch Geometric batch containing graph data

        Returns:
            Updated batch with transformed node features
        """
        x = self.pre_nn(batch.x)

        batch.x = x

        return batch

    @abstractmethod
    def _gnn_layer(self, input_dim: int, output_dim: int) -> nn.Module:
        """
        Create a GNN layer with the specified dimensions.

        Args:
            input_dim: Input dimension for the layer
            output_dim: Output dimension for the layer

        Returns:
            A GNN layer implementation (must be implemented by subclasses)
        """
        pass

    def _build_gnn(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        activation: ActivationType,
        dropout: float,
        layers_num: int,
        act_last: bool,
    ) -> nn.ModuleList:
        """
        Build the GNN layers based on configuration.

        Args:
            input_dim: Input dimension for the first layer
            hidden_dim: Hidden dimension for intermediate layers
            output_dim: Output dimension for the final layer
            activation: Activation function to use
            dropout: Dropout rate
            layers_num: Number of GNN layers
            act_last: Whether to apply activation on the last layer

        Returns:
            ModuleList containing GNN layers
        """

        act_fn = NodeWrapper(PyTorchUtils.activation(activation))

        layers: list[nn.Module] = []

        if self.config.stage_type == StageType.SKIPSUM:
            linears: list[nn.Module] = []
        for n in range(layers_num):
            # Determine dimensions and activation for current layer
            layer_config = self._get_layer_config(
                layer_idx=n,
                total_layers=layers_num,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                act_fn=act_fn,
                act_last=act_last,
            )

            if self.config.stage_type == StageType.SKIPSUM:
                skip_layer: list[nn.Module] = [nn.Linear(layer_config["input_dim"], output_dim)]
                if dropout > 0.0:
                    skip_layer.append(nn.Dropout(dropout))
                linears.append(nn.Sequential(*skip_layer))

            gnn_layer = self._gnn_layer(
                input_dim=layer_config["input_dim"],
                output_dim=layer_config["output_dim"],
            )
            layers_i = [gnn_layer]

            if self.config.batch_norm:
                layers_i.append(NodeWrapper(nn.BatchNorm1d(layer_config["output_dim"])))

            layers_i.append(layer_config["activation"])
            if dropout > 0.0:
                layers_i.append(NodeWrapper(nn.Dropout(dropout)))

            layers.append(nn.Sequential(*layers_i))

        if layers_num == 0:
            layers = [nn.Identity()]

        if self.config.stage_type == StageType.SKIPSUM:
            self.lin_skipsum = nn.ModuleList(linears)

        return nn.ModuleList(layers)

    def _get_layer_config(
        self,
        layer_idx: int,
        total_layers: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        act_fn: NodeWrapper,
        act_last: bool,
    ) -> dict[str, Any]:
        """
        Helper method to determine layer configuration based on layer index

        Returns:
            Dictionary containing input_dim, output_dim, and activation for the layer

        """
        if layer_idx == 0:
            if total_layers == 1 and not act_last:
                return {
                    "input_dim": input_dim,
                    "output_dim": output_dim,
                    "activation": NodeWrapper(nn.Identity()),
                }
            return {
                "input_dim": input_dim,
                "output_dim": hidden_dim,
                "activation": act_fn,
            }
        if layer_idx == (total_layers - 1) and not act_last:
            return {
                "input_dim": hidden_dim,
                "output_dim": hidden_dim,
                "activation": NodeWrapper(nn.Identity()),
            }
        return {"input_dim": hidden_dim, "output_dim": hidden_dim, "activation": act_fn}

    def forward_gnn(self, batch: pygd.Batch, **kwargs) -> pygd.Batch:
        """
        Forward pass through the GNN layers.

        Args:
            batch: PyTorch Geometric batch containing graph data
            **kwargs: Additional arguments to pass to GNN layers

        Returns:
            Updated batch with node features after message passing
        """
        if self.config.num_layers_gnn == 0:
            return batch

        out: torch.Tensor

        for i, layer_i in enumerate(self.gnn):
            if self.lin_skipsum is not None:
                if i == 0:
                    out = self.lin_skipsum[i](batch.x)
                else:
                    out.add_(self.lin_skipsum[i](batch.x))

            batch = layer_i(batch, **kwargs)

        if self.config.stage_type == StageType.SKIPSUM:
            out.add_(batch.x)
            batch.x = out
        return batch

    def post_forward(self, batch: pygd.Batch) -> pygd.Batch:
        """
        Forward pass through the post-processing MLP.

        Args:
            batch: PyTorch Geometric batch with processed node features

        Returns:
            Updated batch with final node representations
        """
        x = self.post_nn(batch.x)
        batch.x = x
        return batch
