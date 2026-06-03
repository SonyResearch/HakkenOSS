import torch

from .base import FactValidator


class NoOpFactValidator(FactValidator):
    """
    No-op validator that doesn't validate anything (always returns False).
    Useful when validation is disabled.
    """

    def validate_batch(self, facts: torch.Tensor) -> torch.Tensor:
        """
        Validate a batch of facts (no-op: always returns False mask).

        Args:
            facts: Facts tensor of shape [N, M]

        Returns:
            Mask of size [N] with all False (no facts are considered positive)
        """
        return torch.zeros(facts.shape[0], device=facts.device, dtype=torch.bool)
