from __future__ import annotations

from typing import Annotated

import torch_geometric.nn as pygnn
from hakken_ml_toolkit.ml_utils.extras import PyTorchUtils
from pydantic import Field
from torch import nn

from gnn.architectures.base import GNNI, GNNConfig


class GINConfig(GNNConfig):
    eps: Annotated[float, Field(description="Initial epsilon value.")] = 0.5
    train_eps: bool = Field(default=False, description="Whether epsilon is a trainable parameter.")
    add_self_loops: bool = Field(
        default=True, description="Whether to add self-loops to the convolution."
    )


class MyGINConv(pygnn.GINConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, batch):
        x = super().forward(x=batch.x, edge_index=batch.edge_index)

        batch.x = x
        return batch


class GIN(GNNI[GINConfig]):
    def __init__(self, config: GINConfig):
        super().__init__(config)

    def _gnn_layer(self, input_dim: int, output_dim: int) -> MyGINConv:
        layers: list[nn.Module] = [nn.Linear(input_dim, output_dim)]
        if self.config.batch_norm:
            layers.append(nn.BatchNorm1d(output_dim))
        layers.append(PyTorchUtils.activation(self.config.activation))
        layers.append(nn.Linear(output_dim, output_dim))

        net = nn.Sequential(*layers)
        return MyGINConv(nn=net, eps=self.config.eps, train_eps=self.config.train_eps, aggr="add")
