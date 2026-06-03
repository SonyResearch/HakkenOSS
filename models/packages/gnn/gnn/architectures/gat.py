from __future__ import annotations

from typing import Annotated

import torch_geometric.data as pygd
import torch_geometric.nn as pygnn
from pydantic import Field

from gnn.architectures.base import GNNI, GNNConfig


class GATConfig(GNNConfig):
    """
    Configuration for Graph Attention Network (GAT).
    """

    heads: Annotated[int, Field(gt=0, description="Number of attention heads. Must be > 0.")] = 1
    edge_dim: Annotated[int, Field(ge=0, description="Dimension of edge features.")] = 0
    add_self_loops: bool = Field(
        default=True, description="Whether to add self-loops to the convolutions."
    )


class MyGATConv(pygnn.GATConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, batch: pygd.Data):
        x = super().forward(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            return_attention_weights=None,
        )

        batch.x = x
        return batch


class GAT(GNNI[GATConfig]):
    def __init__(self, config: GATConfig):
        super().__init__(config)

    def _gnn_layer(self, input_dim: int, output_dim: int) -> MyGATConv:
        if output_dim % self.config.heads != 0:
            msg = f"""
            Output dimension ({output_dim}) must be divisible by 
            the number of heads ({self.config.heads}).
            """
            raise ValueError(msg)
        out_channels = output_dim // self.config.heads

        edge_dim = self.config.edge_dim if self.config.edge_dim > 0 else None
        return MyGATConv(
            in_channels=input_dim,
            out_channels=out_channels,
            heads=self.config.heads,
            concat=True,
            negative_slope=0.2,
            dropout=0.0,
            add_self_loops=self.config.add_self_loops,
            edge_dim=edge_dim,
            fill_value="mean",
            bias=True,
        )
