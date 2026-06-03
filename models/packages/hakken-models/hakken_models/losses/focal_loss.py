import torch
from torch import Tensor
from torch.nn import BCEWithLogitsLoss, Module

from hakken_models.losses.utils import reduce


class FLWithLogitsLoss(Module):
    """Focal Loss with Logits for multi-label classification with optional
        uncertainty regularization.

    Implements focal loss (Lin et al., 2017) [https://arxiv.org/abs/1708.02002]
    combined with binary cross-entropy with logits to address class imbalance in
    multi-label classification tasks. The focal term (1 - p_t)^gamma down-weights
    easy examples and focuses training on hard negatives.

    The loss is computed as:
        FL(p_t) = -(1 - p_t)^gamma * log(p_t)

    where p_t is the model's estimated probability for the true class.

    Optionally includes uncertainty regularization term: -lambda * p * (1 - p)
    which encourages the model to produce uncertain predictions by rewarding
    outputs near 0.5. This term acts as a regularizer that prevents overconfident
    predictions, potentially improving model calibration and reducing overfitting
    by discouraging extreme probability estimates.

    The loss with the regularization term was introduced in Dynamically Weighted
    Balanced (DWB) loss.

    Args:
        weight (Tensor, optional): Manual rescaling weight given to each class.
            If provided, has to be a Tensor of size C (number of classes).
        size_average (bool, optional): Deprecated argument for backward compatibility.
        reduction (str): Specifies the reduction to apply to the output.
            Options: 'none', 'mean', 'sum'. Default: 'mean'.
        pos_weight (Tensor, optional): Weight of positive examples for each class.
            Must be a vector with length equal to the number of classes.
        gamma (float): Focusing parameter that controls the rate at which easy
            examples are down-weighted. Higher gamma puts more focus on hard examples.
            Default: 2.0.
        regularization_coeff (float): Coefficient for uncertainty regularization term.
            When > 0, adds a reward proportional to p * (1 - p) that encourages
            uncertain predictions by reducing the total loss when probabilities
            are near 0.5. This can improve model calibration and prevent overconfidence.
            Higher values lead to stronger uncertainty encouragement.
            Default: 0.0 (no regularization).


    Example:
        >>> criterion = FLWithLogitsLoss(gamma=2.0, reduction='mean')
        >>> logits = torch.randn(3, 5, requires_grad=True)  # 3 samples, 5 classes
        >>> targets = torch.randint(0, 2, (3, 5)).float()   # multi-label targets
        >>> loss = criterion(logits, targets)
        >>> loss.backward()
    """

    def __init__(
        self,
        weight: Tensor | None = None,
        size_average: bool | None = None,
        reduction: str = "mean",
        pos_weight: Tensor | None = None,
        gamma: float = 2.0,
        regularization_coeff: float = 0.0,
    ) -> None:
        if gamma < 0:
            msg = f"gamma must be non-negative, got {gamma}"
            raise ValueError(msg)

        super().__init__()

        self.bce_loss = BCEWithLogitsLoss(
            weight=weight,
            size_average=size_average,
            reduction="none",
            pos_weight=pos_weight,
        )
        self.reduction = reduction
        self.gamma = gamma
        self.regularization_coeff = regularization_coeff

    def forward(
        self,
        logits: Tensor,
        labels: Tensor,
        dim: int | None = None,
    ) -> Tensor:
        """Compute the focal loss with logits.

        Args:
            logits (Tensor): Raw predictions from the model (before sigmoid).
                Shape: (N, C) where N is batch size and C is number of classes.
            labels (Tensor): Ground truth binary labels (0 or 1).
                Shape: (N, C) matching logits shape.

        Returns:
            Tensor: Computed focal loss. Shape depends on reduction.

        Note:
            The focal term (1 - p_t)^gamma reduces loss contribution for well-classified examples:
            - When p_t is high (confident correct prediction), (1 - p_t) is small → less loss
            - When p_t is low (uncertain/incorrect prediction), (1 - p_t) is large → more loss
            - Higher gamma increases this modulation effect

            The uncertainty regularization term p * (1 - p) is maximized when p = 0.5
            (maximum uncertainty) and minimized when p approaches 0 or 1 (confident predictions).
            By subtracting this term, we reduce the total loss when the model is uncertain,
            effectively encouraging uncertain predictions and potentially improving calibration.
        """

        p = torch.sigmoid(logits)
        p_t = p * labels + (1 - p) * (1 - labels)
        label_balance_adjust = (1 - p_t) ** self.gamma

        loss_all = label_balance_adjust * self.bce_loss(logits, labels)
        if self.regularization_coeff > 0.0:
            uncertainty_penalty = self.regularization_coeff * (p * (1 - p))
            loss_all -= uncertainty_penalty

        return reduce(loss_all, self.reduction, dim=dim)
