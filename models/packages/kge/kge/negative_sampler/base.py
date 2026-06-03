from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from hakken_ml_toolkit.ml_base_structures import KnowledgeGraph
from hakken_ml_toolkit.ml_utils.extras import PyTorchUtils

from kge.common.exceptions import NotInitializedError, SplitsError
from kge.common.types import LongTensor2D, LongTensor3D
from kge.negative_sampler.config import NegativeSamplerConfig


class NegativeSamplerI(ABC):
    @abstractmethod
    def __init__(self, config: NegativeSamplerConfig, kg: KnowledgeGraph | None = None):
        self.config = config
        self.kg = kg

        self.filtered = self.config.filter_triples is not None

    def set_up(self, kg: KnowledgeGraph, device: str | torch.device = "cpu") -> None:
        kg.to_device(device)
        self.set_knowledge_graph(kg)

    def get_filtered_triples(self) -> LongTensor2D:
        if self.kg is None:
            msg = "Call set_up(kg) before calling corrupt_batch."
            raise NotInitializedError(msg)

        if self.config.filter_triples is not None:
            sro_batch_list = []
            for split in self.config.filter_triples:
                if split in self.kg.facts_dict:
                    sro_batch = self.kg.facts_dict[split].data
                    sro_batch_list.append(sro_batch)

            return PyTorchUtils.concat_tensors(sro_batch_list, dim=0)

        msg = "Provide with a list of splits"
        raise SplitsError(msg)

    def set_knowledge_graph(self, kg: KnowledgeGraph) -> None:
        self.kg = kg

    @abstractmethod
    def to_device(self, device: str | torch.device) -> None:
        pass

    @abstractmethod
    def corrupt_batch(
        self, sro_batch: LongTensor2D, num_negatives: int | None = None
    ) -> LongTensor3D:
        """
        Generate negative samples by corrupting a batch of subject-relation-object triples.

        Args:
            sro_batch: A tensor of shape (batch_size, 3) containing positive triples
                       in the form of (subject, relation, object) indices.
            num_negatives: Optional; The number of negative samples to generate per
                           positive triple. If provided, this temporarily overrides
                           the sampler's default setting.

        Returns:
            A tensor of shape (batch_size, num_negatives, 3) containing corrupted
            triples where each positive triple has num_negatives corresponding
            negative samples.
        """
        pass
