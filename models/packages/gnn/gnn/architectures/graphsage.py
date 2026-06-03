from __future__ import annotations

import torch_geometric.data as pygd
import torch_geometric.nn as pygnn

from gnn.architectures.base import GNNI, GNNConfig


# TODO: Implement num_sampled_edges_per_hop
class GraphSAGEConfig(GNNConfig):
    num_sampled_edges_per_hop: int = 10


class MyGraphSAGEConv(pygnn.SAGEConv):
    def __init__(self, num_sampled_edges_per_hop: int, *args, **kwargs):
        self.num_sampled_edges_per_hop = num_sampled_edges_per_hop
        super().__init__(*args, **kwargs)

    def forward(self, batch: pygd.Data):
        edge_index = batch.edge_index
        x = super().forward(
            x=batch.x,
            edge_index=edge_index,
        )

        batch.x = x
        return batch


class GraphSAGE(GNNI[GraphSAGEConfig]):
    def __init__(self, config: GraphSAGEConfig):
        super().__init__(config)

    def _gnn_layer(self, input_dim: int, output_dim: int) -> MyGraphSAGEConv:
        return MyGraphSAGEConv(
            num_sampled_edges_per_hop=self.config.num_sampled_edges_per_hop,
            in_channels=input_dim,
            out_channels=output_dim,
            bias=True,
        )
