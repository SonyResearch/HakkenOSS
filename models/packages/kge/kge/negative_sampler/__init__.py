from kge.negative_sampler.base import NegativeSamplerI
from kge.negative_sampler.config import NegativeSamplerConfig
from kge.negative_sampler.uniform import (
    UniformNegativeSampler,
    UniformNegativeSamplerConfig,
)

__all__ = [
    "NegativeSamplerConfig",
    "NegativeSamplerI",
    "UniformNegativeSampler",
    "UniformNegativeSamplerConfig",
]
