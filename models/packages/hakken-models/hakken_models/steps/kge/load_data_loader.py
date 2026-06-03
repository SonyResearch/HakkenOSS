from typing import Annotated, Any

import torch
from torch.utils.data import DataLoader, TensorDataset
from zenml import ArtifactConfig, step

from hakken_models.core.configs.train_common import DataLoaderConfig
from hakken_models.data_loaders.kge import get_negative_data_loader
from hakken_models.datasets.deployment import DatasetDeployment
from hakken_models.negative_samplers import NegativeSampler


def get_data_loader(
    dataset: DatasetDeployment, split_name: str, data_loader_kwargs: dict, **kwargs: Any
) -> DataLoader:
    facts_tensor = dataset.get_facts_tensor(split_name=split_name)

    tensor_dataset = TensorDataset(facts_tensor)

    loader_kwargs = data_loader_kwargs.copy()

    loader_kwargs.update(kwargs)
    return DataLoader(tensor_dataset, **loader_kwargs)


@step(enable_cache=False)
def load_dataloaders_step(
    dataset: DatasetDeployment, data_loader_config: DataLoaderConfig
) -> dict[str, DataLoader]:
    train_loader = get_data_loader(
        dataset=dataset,
        split_name="train",
        data_loader_kwargs=data_loader_config.kwargs,
        shuffle=True,
    )
    val_loader = get_data_loader(
        dataset=dataset,
        split_name="val",
        data_loader_kwargs=data_loader_config.kwargs,
        shuffle=False,
    )

    return {"train": train_loader, "val": val_loader}


@step(enable_cache=True)
def load_dataloader_step(
    dataset: DatasetDeployment,
    negative_sampler: NegativeSampler,
    data_loader_kwargs: dict,
    split_names: list[str],
) -> Annotated[DataLoader, ArtifactConfig(name="{split_name}_loader")]:
    loader_kwargs = data_loader_kwargs.copy()
    facts = []

    for split_name in split_names:
        facts_i = dataset.get_facts_tensor(split_name=split_name)
        facts.append(facts_i)

    facts_all = torch.cat(facts, dim=0)
    tensor_dataset = TensorDataset(facts_all)

    return get_negative_data_loader(
        tensor_dataset, negative_sampler=negative_sampler, **loader_kwargs
    )
