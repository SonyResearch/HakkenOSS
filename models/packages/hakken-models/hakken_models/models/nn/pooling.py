import torch
from torch import Tensor, nn


class AttentionPooling(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(embedding_dim, 1)

    def forward(self, x: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        # x: [B, N, D]. padding_mask [B, N], True = padded (ignore).
        scores = self.score(x)  # [B, N, 1]
        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask.unsqueeze(-1), float("-inf"))
        weights = torch.softmax(scores, dim=1)  # [B, N, 1]
        # Guard against fully-padded sequences where softmax produces NaN.
        weights = torch.nan_to_num(weights, nan=0.0)
        return (x * weights).sum(dim=1)  # [B, D]


class MeanPooling(nn.Module):
    def forward(self, x: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        if padding_mask is None:
            return x.mean(dim=1)
        # Masked mean: sum over valid positions, divide by valid count.
        x_masked = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        count = (~padding_mask).sum(dim=1, keepdim=True).clamp(min=1)
        return x_masked.sum(dim=1) / count
