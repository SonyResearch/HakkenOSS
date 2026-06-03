from torch.utils.data import DataLoader

from hakken_models.registries.base import Registry

from .kg_link_neighbor_loader import KGLinkNeighborLoader
from .temporal_kg_link_neighbor_loader import (
    TemporalKGLinkNeighborLoader,
    corrupt_entity_pairs,
)
from .timestamp_grouped_batch_sampler import TimestampGroupedBatchSampler


class DataLoaderRegistry(Registry[DataLoader]):
    pass


data_loader_registry = DataLoaderRegistry("InitStrategy")


data_loader_registry.register_class(DataLoader)
data_loader_registry.register_class(KGLinkNeighborLoader)
data_loader_registry.register_class(TemporalKGLinkNeighborLoader)

__all__ = [
    "DataLoader",
    "KGLinkNeighborLoader",
    "TemporalKGLinkNeighborLoader",
    "TimestampGroupedBatchSampler",
    "corrupt_entity_pairs",
    "data_loader_registry",
]
