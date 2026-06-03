from typing import Annotated

from zenml import ArtifactConfig, step

from hakken_models.core.configs.train_common import DataLoaderConfig
from hakken_models.core.entities.kg_data import KGData
from hakken_models.core.entities.supervised_dataset import SupervisedDataset
from hakken_models.data_loaders import KGLinkNeighborLoader


@step(enable_cache=False)
def load_thiger_dataloader_step(
    kg_data: KGData,
    sup_dataset: SupervisedDataset,
    data_loader_config: DataLoaderConfig,
    shuffle: bool,
) -> Annotated[KGLinkNeighborLoader, ArtifactConfig(name="{split_name}_loader")]:
    loader_kwargs = data_loader_config.kwargs.copy()

    entity_pairs = sup_dataset.entity_pairs
    relations = sup_dataset.relations

    num_neighbors = loader_kwargs.pop("num_neighbors")
    batch_size = loader_kwargs.pop("batch_size")

    return KGLinkNeighborLoader(
        data=kg_data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        edge_label_index=entity_pairs.t().contiguous(),
        edge_label=relations,
        shuffle=shuffle,
        **loader_kwargs,
    )
