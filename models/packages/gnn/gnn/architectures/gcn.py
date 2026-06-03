from __future__ import annotations

import torch_geometric.nn as pygnn

from gnn.architectures.base import GNNI, GNNConfig


class GCNConfig(GNNConfig):
    improved: bool = False
    add_self_loops: bool = True


class MyGCNConv(pygnn.GCNConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, batch):
        x = super().forward(x=batch.x, edge_index=batch.edge_index, edge_weight=None)

        batch.x = x
        return batch


class GCN(GNNI[GCNConfig]):
    def __init__(self, config: GCNConfig):
        super().__init__(config)

    def _gnn_layer(self, input_dim: int, output_dim: int) -> MyGCNConv:
        return MyGCNConv(
            in_channels=input_dim,
            out_channels=output_dim,
            improved=self.config.improved,
            cached=False,
            add_self_loops=self.config.add_self_loops,
            normalize=True,
            bias=True,
        )
