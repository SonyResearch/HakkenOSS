from hakken_ml_toolkit.ml_utils.extras import FactBatchUtils
from torch import Tensor
from zenml import step

from hakken_models.core.entities import KGData
from hakken_models.datasets.deployment import DatasetDeployment
from hakken_models.models.thiger.data_module import THiGERDataModule


def get_data_bundle(
    dataset: DatasetDeployment, split_name: str, split_names: list[str]
) -> tuple[KGData, Tensor, Tensor]:
    kg_data = dataset.get_kg_data(split_names=split_names)
    facts_pt = dataset.get_facts_tensor(split_name=split_name)

    entity_pairs, relations = FactBatchUtils.to_so_batch_and_relations(
        facts_pt[:, :3], num_relations=dataset.num_relations
    )
    return kg_data, entity_pairs, relations


@step(enable_cache=True)
def load_datamodule_step(
    dataset: DatasetDeployment,
    data_loader_kwargs: dict,
) -> THiGERDataModule:
    loader_kwargs = data_loader_kwargs.copy()
    batch_size = loader_kwargs.pop("batch_size", 32)
    num_neighbors = loader_kwargs.pop("num_neighbors", [128, 128])

    train_kg_data, train_entity_pairs, train_relations = get_data_bundle(
        dataset=dataset, split_name="train", split_names=["train"]
    )

    val_kg_data, val_entity_pairs, val_relations = get_data_bundle(
        dataset=dataset, split_name="val", split_names=["train", "val"]
    )

    return THiGERDataModule(
        train_kg_data=train_kg_data,
        train_entity_pairs=train_entity_pairs,
        train_relations=train_relations,
        val_kg_data=val_kg_data,
        val_entity_pairs=val_entity_pairs,
        val_relations=val_relations,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        **loader_kwargs,
    )
