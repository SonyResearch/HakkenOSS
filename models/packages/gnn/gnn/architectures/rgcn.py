from __future__ import annotations

import torch_geometric.data as pygd
import torch_geometric.nn as pygnn

from gnn.architectures.base import GNNI, GNNConfig


class RGCNConfig(GNNConfig):
    num_relations: int | None = None
    num_bases: int | None = None
    aggr: str = "mean"


class MyRGCNConv(pygnn.RGCNConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, batch: pygd.Data):
        x = super().forward(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_type=batch.edge_type,
        )

        batch.x = x
        return batch


class RGCN(GNNI[RGCNConfig]):
    def __init__(self, config: RGCNConfig):
        super().__init__(config)

    def _gnn_layer(self, input_dim: int, output_dim: int) -> MyRGCNConv:
        return MyRGCNConv(
            in_channels=input_dim,
            out_channels=output_dim,
            num_relations=self.config.num_relations,
            num_bases=self.config.num_bases,
            aggr=self.config.aggr,
        )
