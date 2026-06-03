from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING, Any

import torch
import torch_geometric.data as pygd
import torch_geometric.nn as geom_nn
from hakken_ml_toolkit.ml_utils.constants import ActivationType
from hakken_ml_toolkit.ml_utils.extras import PyTorchUtils
from pydantic import BaseModel
from torch import nn

from gnn.common.constants import PoolingType

if TYPE_CHECKING:
    from gnn.common.domain import FloatTensor2D


class GraphPoolingConfig(BaseModel):
    pool_type: list[PoolingType] = field(
        default_factory=lambda: [PoolingType.MEAN],
    )
    in_channels: int = 256
    activation: ActivationType = ActivationType.IDENTITY
    out_channels: int | None = None
    batch_norm: bool = False


class GraphPooling(nn.Module):
    def __init__(self, config: GraphPoolingConfig):
        super().__init__()
        self.config = config

        self.pool = None
        if PoolingType.GLOBAL_ATT in config.pool_type:  # Global Attention
            raise NotImplementedError()

        self.mlp: nn.Module
        if len(self.config.pool_type) > 1:
            self.mlp = nn.Sequential(
                PyTorchUtils.activation(config.activation),
                nn.Linear(config.in_channels, config.in_channels // len(self.config.pool_type)),
            )
        else:
            self.mlp = nn.Identity()

        self.mlp_out: nn.Module
        if config.out_channels is not None:
            self.mlp_out = nn.Sequential(
                PyTorchUtils.activation(config.activation),
                nn.Linear(config.in_channels, config.out_channels),
            )
        else:
            self.mlp_out = nn.Identity()

        self.bn: nn.BatchNorm1d
        if config.batch_norm:
            self.bn = nn.BatchNorm1d(num_features=config.in_channels)

    def forward(self, batch: pygd.Data) -> FloatTensor2D:
        batch_ = batch.batch
        x = batch.x
        return self.forward_from_tensors(x=x, batch=batch_)

    def forward_from_tensors(
        self, x: torch.FloatTensor, batch: torch.Tensor
    ) -> FloatTensor2D | Any:
        z_global = []

        if PoolingType.MEAN in self.config.pool_type:
            z_global_i = geom_nn.global_mean_pool(x=x, batch=batch)
            z_global.append(z_global_i)
        if PoolingType.MAX in self.config.pool_type:
            z_global_i = geom_nn.global_max_pool(x=x, batch=batch)
            z_global.append(z_global_i)
        if PoolingType.ADD in self.config.pool_type:
            z_global_i = geom_nn.global_add_pool(x=x, batch=batch)
            z_global.append(z_global_i)
        if self.pool is not None:
            if isinstance(self.pool, geom_nn.TopKPooling):
                outputs = self.pool(x=x, edge_index=batch.edge_index, edge_attr=batch.edge_attr)
                z_global_i = outputs[0]
            else:
                z_global_i = self.pool(x=x, batch=batch)

            z_global.append(z_global_i)

        if len(z_global) > 1:
            z_global_ = []
            for z_i in z_global:
                z_global_.append(self.mlp(z_i))
            out = torch.cat(z_global_, dim=1)
        else:
            z_pool = torch.cat(z_global, dim=1)

            out = self.mlp(z_pool)

        out = self.mlp_out(out)

        if self.config.batch_norm:
            out = self.bn(out)

        return out
