import torch

from hakken_models.core.constants import FactComponent

from .base import NegativeSampler


class UniformNegativeSampler(NegativeSampler):
    """
    Uniform negative sampler that corrupts facts by randomly replacing
    components (subject, relation, or object) with uniformly sampled values.
    """

    def _corrupt_facts_once(self, facts: torch.Tensor, num_negatives: int = 1) -> torch.Tensor:
        """
        Generate corrupted facts using uniform random sampling.
        Fully vectorized implementation for maximum performance.

        Args:
            facts: Input facts of shape [num_facts, M] where M>=3.
                   Assumes format: [subject_idx, relation_idx, object_idx, ...]
            num_negatives: Number of negative samples per fact.

        Returns:
            Corrupted facts of shape [num_facts, num_negatives, M]
        """
        num_facts, _ = facts.shape
        device = facts.device
        dtype = facts.dtype

        corrupted_facts = facts.unsqueeze(1).expand(-1, num_negatives, -1).clone()

        num_scheme_components = len(self.corruption_scheme)
        corruption_choice = torch.randint(
            0, num_scheme_components, (num_facts, num_negatives), device=device
        )

        random_entities = torch.randint(
            0, self.num_entities, (num_facts, num_negatives), device=device, dtype=dtype
        )

        if self.num_relations is not None:
            random_relations = torch.randint(
                0, self.num_relations, (num_facts, num_negatives), device=device, dtype=dtype
            )

        for i, component in enumerate(self.corruption_scheme):
            mask = corruption_choice == i

            indices = torch.nonzero(mask, as_tuple=False)

            if len(indices) == 0:
                continue

            fact_indices = indices[:, 0]
            neg_indices = indices[:, 1]

            if component == FactComponent.SUBJECT:
                corrupted_facts[fact_indices, neg_indices, 0] = random_entities[mask]
            elif component == FactComponent.RELATION:
                corrupted_facts[fact_indices, neg_indices, 1] = random_relations[mask]
            elif component == FactComponent.OBJECT:
                corrupted_facts[fact_indices, neg_indices, 2] = random_entities[mask]

        return corrupted_facts
