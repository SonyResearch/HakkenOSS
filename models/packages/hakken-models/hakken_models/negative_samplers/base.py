from abc import ABC, abstractmethod

import torch

from hakken_models.core.constants import FactComponent
from hakken_models.fact_validator import FactValidator


class NegativeSampler(ABC):
    def __init__(
        self,
        num_entities: int,
        num_relations: int | None = None,
        corruption_scheme: list[FactComponent] | None = None,
        fact_validator: FactValidator | None = None,
    ) -> None:
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.corruption_scheme = (
            [FactComponent.SUBJECT, FactComponent.OBJECT]
            if corruption_scheme is None
            else corruption_scheme
        )
        self.fact_validator = fact_validator

    @abstractmethod
    def _corrupt_facts_once(self, facts: torch.Tensor, num_negatives: int = 1) -> torch.Tensor:
        """
        Generate corrupted facts without validation retry logic.
        This is the core corruption strategy that subclasses implement.

        Args:
            facts: Input facts of shape [num_facts, M] where M>=3.
            num_negatives: Number of negative samples per fact.

        Returns:
            Corrupted facts of shape [num_facts, num_negatives, M]
        """
        pass

    def corrupt_facts(
        self, facts: torch.Tensor, num_negatives: int = 1, num_attempts: int = 1
    ) -> torch.Tensor:
        """
        Corrupt facts to generate negative samples with validation retry logic.

        Args:
            facts: Input facts of shape [num_facts, M] where M>=3.
            num_negatives: Number of negative samples per fact.
            num_attempts: Number of attempts to generate valid negative samples.

        Returns:
            Corrupted facts of shape [num_facts, num_negatives, M]
        """
        corrupted_facts = self._corrupt_facts_once(facts, num_negatives)
        if self.fact_validator is None or num_attempts <= 1:
            return corrupted_facts

        num_facts, num_negatives, num_components = corrupted_facts.shape

        for _attempt in range(1, num_attempts):
            flat_corrupted = corrupted_facts.view(-1, num_components)
            is_positive = self.fact_validator.validate_batch(flat_corrupted)
            is_positive = is_positive.view(num_facts, num_negatives)

            if not is_positive.any():
                break

            # Get indices of invalid (positive) samples: [fact_idx, neg_idx] pairs
            invalid_indices = torch.nonzero(is_positive, as_tuple=False)

            if len(invalid_indices) == 0:
                break

            fact_indices = invalid_indices[:, 0]  # Shape: [num_invalid]
            neg_indices = invalid_indices[:, 1]  # Shape: [num_invalid]
            original_facts = facts[fact_indices]  # Shape: [num_invalid, num_components]
            new_corrupted = self._corrupt_facts_once(original_facts, num_negatives=1)
            corrupted_facts[fact_indices, neg_indices] = new_corrupted.squeeze(1)

        return corrupted_facts
