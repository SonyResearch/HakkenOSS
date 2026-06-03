"""Step to load SeGAL evaluation dataloader."""

from typing import Annotated

from zenml import ArtifactConfig, step

from hakken_models.core.configs.train_common import DataLoaderConfig
from hakken_models.core.entities.kg_data import KGData
from hakken_models.data_loaders import TemporalKGLinkNeighborLoader
from hakken_models.datasets.deployment import DatasetDeployment

_COL_S = 0
_COL_R = 1
_COL_O = 2
_COL_T = 3


@step(enable_cache=False)
def load_segal_dataloader_step(
    kg_data: KGData,
    dataset: DatasetDeployment,
    split_name: str,
    data_loader_config: DataLoaderConfig,
    num_negatives: int = 32,
    shuffle: bool = False,
) -> Annotated[TemporalKGLinkNeighborLoader, ArtifactConfig(name="{split_name}_loader")]:
    """Build a TemporalKGLinkNeighborLoader for SeGAL evaluation.

    Args:
        kg_data: Full KG context (e.g. train+val for evaluating on val).
        dataset: Dataset deployment for target split facts.
        split_name: Split to evaluate on (train, val, or test).
        data_loader_config: DataLoader configuration.
        num_negatives: Number of corrupted entity pairs per positive.
        shuffle: Whether to shuffle batches.

    Returns:
        TemporalKGLinkNeighborLoader for evaluation.
    """
    loader_kwargs = data_loader_config.kwargs.copy()
    num_neighbors = loader_kwargs.pop("num_neighbors", [128, 128])
    batch_size = loader_kwargs.pop("batch_size", 128)

    facts = dataset.get_facts_tensor(split_name=split_name)

    if facts.shape[1] < 4:
        raise ValueError(
            f"SeGAL requires temporal facts (s, r, o, t). "
            f"Got {facts.shape[1]} columns for split '{split_name}'."
        )

    entity_pairs = facts[:, [_COL_S, _COL_O]]
    relations = facts[:, _COL_R]
    timestamps = facts[:, _COL_T].float()

    relation_labels = None
    if dataset.has_relation_labels and split_name in ("train", "val"):
        relation_labels = dataset.get_relation_labels_tensor(split_name)

    return TemporalKGLinkNeighborLoader(
        data=kg_data,
        num_neighbors=num_neighbors,
        edge_label_index=entity_pairs.t().contiguous(),
        edge_label=relations,
        target_timestamps=timestamps,
        num_negatives=num_negatives,
        target_relation_labels=relation_labels,
        batch_size=batch_size,
        shuffle=shuffle,
        **loader_kwargs,
    )
