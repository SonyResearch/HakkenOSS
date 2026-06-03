import torch
import torch.nn.functional as torch_f
from torch import nn

from hakken_models.losses.utils import reduce


class NSSALoss(nn.Module):
    """
    Self-Adversarial Negative Sampling Loss (NSSA / RotatE-style loss).

    Matches equation (6) from:
    Zhiqing Sun et al. (2019)
    "RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space"
    https://arxiv.org/abs/1902.10197

    Scoring convention: higher score = more plausible triple

    Args:
        margin: Margin hyperparameter. Common values in RotatE: 6.0 - 12.0
        adversarial_temperature: Temperature for self-adversarial weighting.
                                 Common: 0.5 - 1.0 (higher indicates harder negatives weighted more)
        reduction: 'mean' | 'sum' | 'none'
    """

    def __init__(
        self, margin: float = 9.0, adversarial_temperature: float = 1.0, reduction: str = "mean"
    ):
        super().__init__()
        self.margin = margin
        self.adversarial_temperature = adversarial_temperature
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"Invalid reduction: {reduction}")
        self.reduction = reduction

    def forward(
        self,
        pos_score: torch.Tensor,
        neg_score: torch.Tensor,
        *args,
        dim: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute the RotatE self-adversarial negative sampling loss.

        Args:
            pos_score: Scores of positive triples (higher = better, shape e.g. [batch])
            neg_score: Scores of negative triples (higher = better, shape e.g. [batch, num_neg])

        Returns:
            Scalar loss (after reduction) or per-example loss if reduction='none'
        """
        # ── Positive loss term ───────────────────────────────────────
        positive_loss = torch_f.softplus(-pos_score)  # [batch_size]

        # ── Self-adversarial weights for negatives ───────────────────
        exp_weights = torch.exp(neg_score / self.adversarial_temperature)  # [batch, n_neg]
        weights = torch_f.softmax(exp_weights, dim=-1)  # [batch, n_neg]
        weights = weights.detach()  # stop gradient (as in paper)

        # ── Weighted negative loss term ──────────────────────────────
        negative_term = torch_f.softplus(neg_score)  # [batch, n_neg]
        weighted_negative_loss = weights * negative_term
        negative_loss = weighted_negative_loss.sum(dim=-1)  # [batch_size]

        # ── Total loss ───────────────────────────────────────────────
        loss = positive_loss + negative_loss

        return reduce(loss, self.reduction, dim=dim)
