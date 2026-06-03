"""Relational GraphSAGE — GraphSAGE variant with edge-feature support.

Standard GraphSAGE ignores edge attributes entirely: messages are just
neighbour node features.  :class:`RelationalSAGEConv` fuses edge features
(e.g. relation embeddings + temporal encodings) into every message *before*
aggregation, giving the model the same relational expressiveness that
:pyg:`GATConv` gets through its ``edge_dim`` parameter.

Message function
----------------

.. math::

    \\mathbf{m}_{j \\to i} = \\mathbf{W}_{\\text{msg}}
    \\bigl[\\mathbf{x}_j \\,\\|\\, \\mathbf{e}_{j \\to i}\\bigr]

Update rule
-----------

.. math::

    \\mathbf{x}_i^{\\prime} =
    \\mathbf{W}_{\\text{agg}} \\;
    \\operatorname{AGGREGATE}_{j \\in \\mathcal{N}(i)}
    \\bigl(\\mathbf{m}_{j \\to i}\\bigr)
    \\;+\\; \\mathbf{W}_{\\text{root}} \\, \\mathbf{x}_i
"""

from __future__ import annotations

from typing import Final

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.models.basic_gnn import BasicGNN
from torch_geometric.typing import Adj, OptPairTensor, Size


class RelationalSAGEConv(MessagePassing):
    """SAGEConv that incorporates edge features into neighbour messages.

    When ``edge_dim`` is provided, each message is computed as::

        msg_j = W_msg([x_j || edge_attr_j])

    When ``edge_dim is None``, it falls back to standard SAGEConv behaviour
    (identity message).

    Args:
        in_channels: Size of source node features (or tuple for bipartite).
        out_channels: Size of output node features.
        edge_dim: Size of edge feature vectors.  ``None`` disables edge
            feature fusion (vanilla SAGEConv).
        aggr: Aggregation scheme (``"mean"``, ``"max"``, …).
        normalize: L2-normalize output features.
        root_weight: Add a learned self-loop transformation.
        project: Apply a pre-projection to source features before messaging
            (Eq. 3 in Hamilton et al.).
        bias: Learn an additive bias in the aggregation linear layer.
    """

    def __init__(
        self,
        in_channels: int | tuple[int, int],
        out_channels: int,
        edge_dim: int | None = None,
        aggr: str = "mean",
        normalize: bool = False,
        root_weight: bool = True,
        project: bool = False,
        bias: bool = True,
        **kwargs,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.normalize = normalize
        self.root_weight = root_weight
        self.project = project

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        super().__init__(aggr=aggr, **kwargs)

        if self.project:
            self.lin_proj = Linear(in_channels[0], in_channels[0], bias=True)

        msg_in = in_channels[0] + edge_dim if edge_dim is not None else in_channels[0]
        self.lin_msg = Linear(msg_in, out_channels, bias=bias)

        if self.root_weight:
            self.lin_root = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if self.project:
            self.lin_proj.reset_parameters()
        self.lin_msg.reset_parameters()
        if self.root_weight:
            self.lin_root.reset_parameters()

    def forward(
        self,
        x: Tensor | OptPairTensor,
        edge_index: Adj,
        edge_attr: Tensor | None = None,
        size: Size = None,
    ) -> Tensor:
        if isinstance(x, Tensor):
            x = (x, x)

        if self.project:
            x = (self.lin_proj(x[0]).relu(), x[1])

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_root = x[1]
        if self.root_weight and x_root is not None:
            out = out + self.lin_root(x_root)

        if self.normalize:
            out = F.normalize(out, p=2.0, dim=-1)

        return out

    def message(self, x_j: Tensor, edge_attr: Tensor | None) -> Tensor:
        if edge_attr is not None:
            return self.lin_msg(torch.cat([x_j, edge_attr], dim=-1))
        return self.lin_msg(x_j)


class RelationalGraphSAGE(BasicGNN):
    """Multi-layer GraphSAGE with edge-feature support.

    Drop-in replacement for :pyg:`GraphSAGE` that passes ``edge_attr``
    through every layer via :class:`RelationalSAGEConv`.

    Accepts the same constructor arguments as :pyg:`BasicGNN` plus any
    extra kwargs forwarded to :class:`RelationalSAGEConv` (notably
    ``edge_dim``).
    """

    supports_edge_weight: Final[bool] = False
    supports_edge_attr: Final[bool] = True

    def init_conv(
        self,
        in_channels: int | tuple[int, int],
        out_channels: int,
        **kwargs,
    ) -> MessagePassing:
        return RelationalSAGEConv(in_channels, out_channels, **kwargs)
