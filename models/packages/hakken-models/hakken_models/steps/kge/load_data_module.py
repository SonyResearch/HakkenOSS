from loguru import logger
from zenml import step

from hakken_models.datasets.deployment import DatasetDeployment
from hakken_models.models.kge.data_module import KGEDataModule
from hakken_models.negative_samplers import NegativeSampler


@step(enable_cache=True)
def load_datamodule_step(
    dataset: DatasetDeployment,
    negative_sampler: NegativeSampler,
    data_loader_kwargs: dict,
) -> KGEDataModule:
    loader_kwargs = data_loader_kwargs.copy()
    batch_size = loader_kwargs.pop("batch_size", 32)
    train_facts = dataset.get_facts_tensor(split_name="train")
    val_facts = dataset.get_facts_tensor(split_name="val")

    data_module_kwargs: dict = {
        "train_facts": train_facts,
        "val_facts": val_facts,
        "negative_sampler": negative_sampler,
        "val_num_negatives": 5_000,
        "batch_size": batch_size,
        **loader_kwargs,
    }
    data_module_kwargs["num_relations"] = dataset.num_relations
    logger.info(
        "Using pair-aggregated relation labels from train/val facts "
        f"(num_relations={dataset.num_relations})"
    )

    return KGEDataModule(**data_module_kwargs)
