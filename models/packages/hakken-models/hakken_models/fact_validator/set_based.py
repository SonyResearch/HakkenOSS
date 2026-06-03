import torch

from .base import FactValidator


class SetFactValidator(FactValidator):
    """
    Validate facts using a set of positive facts for fast lookup.
    """

    def __init__(
        self,
        positive_facts: torch.Tensor,
        fact_key_length: int = 3,
        strict: bool = True,
    ):
        """
        Args:
            positive_facts: All positive facts of shape [num_positives, M]
            fact_key_length: Number of columns to use for comparison (default: 3 for s, r, o)
            strict: If True, exact match required. If False, allows approximate matching
        """
        self.fact_key_length = fact_key_length
        self.strict = strict

        self._positive_set = {
            tuple(fact[:fact_key_length].cpu().tolist()) for fact in positive_facts
        }

    def validate_batch(self, facts: torch.Tensor) -> torch.Tensor:
        """
        Validate a batch of facts, checking if they are positive (in the data).

        Args:
            facts: Facts tensor of shape [N, M]

        Returns:
            Mask of size [N] with True if the fact is positive (one that is in the data)
        """
        keys = [
            tuple(facts[i, : self.fact_key_length].cpu().tolist()) for i in range(facts.shape[0])
        ]

        return torch.tensor(
            [key in self._positive_set for key in keys],
            device=facts.device,
            dtype=torch.bool,
        )
