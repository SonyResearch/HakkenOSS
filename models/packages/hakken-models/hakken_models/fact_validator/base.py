from abc import ABC, abstractmethod

import torch


class FactValidator(ABC):
    """
    Base class for validating facts (checking if facts are positive/in the data).
    """

    @abstractmethod
    def validate_batch(self, facts: torch.Tensor) -> torch.Tensor:
        """
        Validate a batch of facts, checking if they are positive (in the data).

        Args:
            facts: Facts tensor of shape [N, M]

        Returns:
            Mask of size [N] with True if the fact is positive (one that is in the data)
        """
        pass
