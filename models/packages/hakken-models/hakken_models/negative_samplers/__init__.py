from hakken_models.registries.base import Registry

from .base import NegativeSampler
from .uniform import UniformNegativeSampler


class NegativeSamplerRegistry(Registry[NegativeSampler]):
    pass


neg_sampler_registry = NegativeSamplerRegistry("NegativeSampler")


neg_sampler_registry.register_class(UniformNegativeSampler)

__all__ = ["NegativeSampler", "UniformNegativeSampler", "neg_sampler_registry"]
