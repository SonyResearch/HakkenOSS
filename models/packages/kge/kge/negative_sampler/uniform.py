from __future__ import annotations

import torch
from hakken_ml_toolkit.ml_base_structures import KnowledgeGraph
from pykeen.sampling import BasicNegativeSampler

from kge.common.exceptions import NotInitializedError, WrongCorruptionSchemeError
from kge.common.types import LongTensor2D, LongTensor3D
from kge.negative_sampler.base import NegativeSamplerConfig, NegativeSamplerI


class UniformNegativeSamplerConfig(NegativeSamplerConfig):
    pass


class UniformNegativeSampler(NegativeSamplerI):
    def __init__(self, config: UniformNegativeSamplerConfig):
        super().__init__(config)
        self.config = config

        self._sampler: BasicNegativeSampler | None = None

    @property
    def sampler(self) -> BasicNegativeSampler:
        if self._sampler is None:
            msg = "Sampler not initialized. Call set_up() first."
            raise NotInitializedError(msg)
        return self._sampler

    def set_up(self, kg: KnowledgeGraph, device: str | torch.device = "cpu") -> None:
        """
        Configures BasicNegativeSampler based on corruption scheme settings
        and filtering preferences. See PyKEEN documentation for details:
        https://pykeen.readthedocs.io/en/v1.11.1/api/pykeen.sampling.BasicNegativeSampler.html

        Raises:
            WrongCorruptionSchemeError: If no valid corruption scheme specified
        """
        super().set_up(kg, device)
        corruption_scheme = []
        if "object" in self.config.corruption_scheme:
            corruption_scheme.append("tail")
        if "subject" in self.config.corruption_scheme:
            corruption_scheme.append("head")
        if "relation" in self.config.corruption_scheme:
            corruption_scheme.append("relation")
        if len(corruption_scheme) == 0:
            raise WrongCorruptionSchemeError()

        if self.filtered:
            mapped_triples = self.get_filtered_triples()
            filterer = "bloom"
            filterer_kwargs = {"error_rate": 0.001}
        else:
            mapped_triples = None
            filterer = None
            filterer_kwargs = None

        # Documentation: https://pykeen.readthedocs.io/en/v1.11.1/api/pykeen.sampling.BasicNegativeSampler.html
        self._sampler = BasicNegativeSampler(
            corruption_scheme=corruption_scheme,  # type: ignore
            num_entities=kg.num_entities,
            num_relations=kg.num_relations,
            num_negs_per_pos=self.config.num_negatives,
            mapped_triples=mapped_triples.to("cpu"),
            filtered=self.filtered,
            filterer=filterer,
            filterer_kwargs=filterer_kwargs,
        )
        self.to_device(device)

    def to_device(self, device: str | torch.device):
        self.sampler.filterer.mersenne = self.sampler.filterer.mersenne.to(device)  # type: ignore
        self.sampler.filterer.bit_array = self.sampler.filterer.bit_array.to(device)  # type: ignore

    def corrupt_batch(
        self, sro_batch: LongTensor2D, num_negatives: int | None = None
    ) -> LongTensor3D:
        if self.kg is None:
            raise NotInitializedError()

        if num_negatives is not None:
            num_negatives_original = self.sampler.num_negs_per_pos
            self.sampler.num_negs_per_pos = num_negatives
        sro_negative, _valid_negatives = self.sampler.sample(sro_batch)

        if num_negatives is not None:
            self.sampler.num_negs_per_pos = num_negatives_original

        return sro_negative
