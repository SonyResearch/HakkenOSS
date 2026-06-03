import torch

from .base import NegativeSampler


class BernoulliNegativeSamper(NegativeSampler):
    def _corrupt_facts_once(self, facts: torch.Tensor, num_negatives: int = 1) -> torch.Tensor:
        raise NotImplementedError()
