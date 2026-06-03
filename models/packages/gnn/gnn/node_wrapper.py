from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

if TYPE_CHECKING:
    import torch_geometric.data as pygd


class NodeWrapper(nn.Module):
    def __init__(self, layer: nn.Module):
        super().__init__()

        self.layer = layer

    def forward(self, batch: pygd.Data, **kwargs):
        batch.x = self.layer(batch.x, **kwargs)
        return batch
