import pytest

from hakken_models.negative_samplers import UniformNegativeSampler

from .base import BaseNegativeSamplerTests


class TestUniformNegativeSampler(BaseNegativeSamplerTests):
    __test__ = True

    @pytest.fixture
    def sampler(self, sampler_config: dict) -> UniformNegativeSampler:
        """Return a UniformNegativeSampler instance with specified parameters."""
        return UniformNegativeSampler(
            num_entities=sampler_config["num_entities"],
            num_relations=sampler_config.get("num_relations"),
            corruption_scheme=sampler_config.get("corruption_scheme"),
            fact_validator=sampler_config.get("fact_validator"),
        )
