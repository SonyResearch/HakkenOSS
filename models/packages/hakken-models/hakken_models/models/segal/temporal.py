"""Continuous temporal encoding for open-vocabulary year timestamps."""

import math

import torch
from torch import Tensor, nn


class TemporalEncoder(nn.Module):
    """Hybrid sinusoidal + learnable-linear temporal encoder.

    Produces embeddings for continuous year values that generalise to unseen
    (including future) timestamps.  The linear component captures monotonic
    trends and ordering; the sinusoidal component captures relative distances
    and optional cyclic patterns.

    Normalisation statistics (mean / std of training timestamps) must be set
    via :meth:`set_normalization` before the first forward pass.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        learnable_frequencies: bool = True,
        num_sinusoidal: int | None = None,
    ):
        super().__init__()

        if num_sinusoidal is None:
            num_sinusoidal = embedding_dim // 2
        linear_dim = embedding_dim - num_sinusoidal

        self.linear = nn.Linear(1, linear_dim)

        if learnable_frequencies:
            self.frequencies = nn.Parameter(torch.randn(num_sinusoidal) * 0.02)
            self.phases = nn.Parameter(torch.zeros(num_sinusoidal))
        else:
            freqs = torch.exp(torch.linspace(math.log(0.1), math.log(100.0), num_sinusoidal))
            self.register_buffer("frequencies", freqs)
            self.register_buffer("phases", torch.zeros(num_sinusoidal))

        self.norm = nn.LayerNorm(embedding_dim)

        self.register_buffer("t_mean", torch.tensor(0.0))
        self.register_buffer("t_std", torch.tensor(1.0))

    def set_normalization(self, mean: float, std: float) -> None:
        """Store training-set timestamp statistics for normalisation."""
        self.t_mean.fill_(mean)
        self.t_std.fill_(max(std, 1e-6))

    def forward(self, t: Tensor) -> Tensor:
        """Encode year values into dense temporal embeddings.

        Args:
            t: ``[...]`` float tensor of year values.

        Returns:
            ``[..., embedding_dim]`` temporal embeddings.
        """
        t_norm = (t.float() - self.t_mean) / self.t_std
        t_unsq = t_norm.unsqueeze(-1)

        linear_part = self.linear(t_unsq)
        angles = t_unsq * self.frequencies + self.phases
        periodic_part = torch.sin(angles)

        return self.norm(torch.cat([linear_part, periodic_part], dim=-1))
