from zenml import step

from hakken_models.datasets.deployment import DatasetDeployment
from hakken_models.models.segal.data_module import SeGALDataModule


@step
def load_segal_datamodule_step(
    dataset: DatasetDeployment,
    data_loader_kwargs: dict,
    num_negatives: int = 32,
    num_negatives_val: int | None = None,
    add_reverse_edges: bool = True,
) -> SeGALDataModule:
    """Build a SeGALDataModule from a dataset deployment."""
    loader_kwargs = data_loader_kwargs.copy()
    batch_size = loader_kwargs.pop("batch_size", 32)
    num_neighbors = loader_kwargs.pop("num_neighbors", [128, 128])

    train_facts = dataset.get_facts_tensor(split_name="train")
    val_facts = dataset.get_facts_tensor(split_name="val")

    train_relation_labels = None
    val_relation_labels = None
    if dataset.has_relation_labels:
        train_relation_labels = dataset.get_relation_labels_tensor("train")
        val_relation_labels = dataset.get_relation_labels_tensor("val")

    return SeGALDataModule(
        train_facts=train_facts,
        val_facts=val_facts,
        num_nodes=dataset.num_entities,
        train_relation_labels=train_relation_labels,
        val_relation_labels=val_relation_labels,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        num_negatives=num_negatives,
        num_negatives_val=num_negatives_val,
        add_reverse_edges=add_reverse_edges,
        **loader_kwargs,
    )
