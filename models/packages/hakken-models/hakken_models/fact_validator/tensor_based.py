import torch

from .base import FactValidator


class TensorFactValidator(FactValidator):
    """
    GPU-optimized validator using tensor operations for fast batch validation.

    This validator stores positive facts as a tensor and uses broadcasting
    and efficient tensor operations to validate batches without Python loops.
    """

    def __init__(
        self,
        positive_facts: torch.Tensor,
        fact_key_length: int = 3,
        device: torch.device | None = None,
        batch_size: int = 10000,
    ):
        """
        Args:
            positive_facts: All positive facts of shape [num_positives, M]
            fact_key_length: Number of columns to use for comparison (default: 3 for s, r, o)
            device: Device to store positive facts on. If None, uses device of positive_facts.
            batch_size: Maximum number of facts to process at once (default: 10000)
        """
        self.fact_key_length = fact_key_length
        self.batch_size = batch_size

        if device is None:
            device = positive_facts.device

        self._positive_keys = positive_facts[:, :fact_key_length].to(device).long()
        self.device = device

    def validate_batch(self, facts: torch.Tensor) -> torch.Tensor:
        """
        Validate a batch of facts using efficient GPU tensor operations.
        Processes facts in chunks to avoid memory issues with large batches.

        Args:
            facts: Facts tensor of shape [N, M]

        Returns:
            Mask of size [N] with True if the fact is positive (one that is in the data)
        """
        facts = facts.to(self.device)
        num_facts = facts.shape[0]

        # Process in batches if needed
        if self.batch_size >= num_facts:
            return self._validate_chunk(facts)

        # Process in chunks
        results = []
        for i in range(0, num_facts, self.batch_size):
            end_idx = min(i + self.batch_size, num_facts)
            chunk = facts[i:end_idx]
            chunk_result = self._validate_chunk(chunk)
            results.append(chunk_result)

        return torch.cat(results, dim=0)

    def _validate_chunk(self, facts: torch.Tensor) -> torch.Tensor:
        """
        Validate a chunk of facts (internal method for processing sub-batches).

        Args:
            facts: Facts tensor of shape [N, M] where N <= batch_size

        Returns:
            Mask of size [N] with True if the fact is positive
        """
        fact_keys = facts[:, : self.fact_key_length]

        fact_keys_expanded = fact_keys.unsqueeze(1)  # [N, 1, fact_key_length]
        positive_keys_expanded = self._positive_keys.unsqueeze(
            0
        )  # [1, num_positives, fact_key_length]

        # Element-wise equality: [N, num_positives, fact_key_length]
        matches = fact_keys_expanded == positive_keys_expanded

        # Check if all key components match: [N, num_positives]
        all_match = torch.all(matches, dim=-1)

        # Check if any positive fact matches: [N]
        return torch.any(all_match, dim=-1)
