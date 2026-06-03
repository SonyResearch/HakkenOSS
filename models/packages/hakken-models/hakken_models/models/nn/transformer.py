"""Transformer modules for sequence and temporal aggregation."""

import math
from typing import cast

import torch
from strenum import StrEnum
from torch import Tensor, nn

from hakken_models.registries.base import Registry

from .pooling import AttentionPooling, MeanPooling

_INIT_STD = 0.02


class AggregationType(StrEnum):
    """How to aggregate the sequence dimension [B, N, D] -> [B, D]."""

    MEAN = "mean"
    ATTENTION = "attention"
    CLS_TOKEN = "cls_token"


class Transformer(nn.Module):
    """Single configurable transformer: [B, N, D] -> [B, D].

    Configure via __init__:
    - aggregation: "mean", "attention", or "cls_token" (how to get [B, D] from encoder output).
    - use_pos_encoding: add learnable positional encoding.
    - norm_first: if True, use Pre-LN (LayerNorm before attention/FFN); else Post-LN.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_pos_encoding: bool = True,
        aggregation: AggregationType | str = AggregationType.MEAN,
        max_seq_len: int = 1000,
        norm_first: bool = False,
    ):
        super().__init__()
        aggregation = AggregationType(aggregation) if isinstance(aggregation, str) else aggregation

        self.aggregation = aggregation
        self.use_pos_encoding = use_pos_encoding
        self._embedding_dim = embedding_dim

        # FFN hidden size: 4x d_model follows "Attention is All You Need" (Vaswani et al.)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=4 * embedding_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=norm_first,
        )
        # Pre-LN leaves the residual stream unnormalized after the last layer;
        # a final LayerNorm is required (Xiong et al., 2020).
        encoder_norm = nn.LayerNorm(embedding_dim) if norm_first else None
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, norm=encoder_norm
        )

        if aggregation == AggregationType.CLS_TOKEN:
            self.cls_token = nn.Parameter(torch.empty(1, 1, embedding_dim))
            nn.init.normal_(self.cls_token, std=_INIT_STD)
            self.pool = None
        else:
            self.register_parameter("cls_token", None)
            if aggregation == AggregationType.ATTENTION:
                self.pool = AttentionPooling(embedding_dim)
            elif aggregation == AggregationType.MEAN:
                self.pool = MeanPooling()
            else:
                raise ValueError(f"Invalid aggregation: {aggregation}")

        if use_pos_encoding:
            self.pos_encoding = nn.Parameter(torch.empty(max_seq_len, embedding_dim))
            nn.init.normal_(self.pos_encoding, std=_INIT_STD)
        else:
            self.register_parameter("pos_encoding", None)

        self.embed_dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        """Input x: [B, N, D]. Optional padding_mask [B, N] with True = padded (ignored).

        Output: [B, D].
        """
        b, n, d = x.shape

        # Scale embeddings so content signal is not dominated by positional encoding
        # (Vaswani et al., "Attention is All You Need", Section 3.4).
        x = x * math.sqrt(d)

        if self.use_pos_encoding and self.pos_encoding is not None:
            if n > self.pos_encoding.size(0):
                raise ValueError(
                    f"Sequence length {n} exceeds max_seq_len {self.pos_encoding.size(0)}"
                )
            x = x + self.pos_encoding[:n].unsqueeze(0)

        x = self.embed_dropout(x)

        if self.aggregation == AggregationType.CLS_TOKEN and self.cls_token is not None:
            cls_tokens = self.cls_token.expand(b, 1, d)
            x = torch.cat([cls_tokens, x], dim=1)
            if padding_mask is not None:
                key_padding_mask = torch.cat(
                    [torch.zeros(b, 1, dtype=torch.bool, device=x.device), padding_mask],
                    dim=1,
                )
            else:
                key_padding_mask = None
            x = self.encoder(x, src_key_padding_mask=key_padding_mask)
            return cast("Tensor", x[:, 0, :])

        x = self.encoder(x, src_key_padding_mask=padding_mask)
        if self.pool is None:
            raise RuntimeError(
                f"Pooling layer is None for aggregation={self.aggregation}; "
                "this should never happen."
            )
        return self.pool(x, padding_mask)


class TransformerRegistry(Registry[Transformer]):
    """Registry for Transformer models."""


tx_registry = TransformerRegistry("Transformer")
tx_registry.register_class(Transformer)

__all__ = [
    "AggregationType",
    "Transformer",
    "TransformerRegistry",
    "tx_registry",
]
