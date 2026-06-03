from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel
from torch.utils.data import DataLoader, Dataset, TensorDataset

from datasets.common.exceptions import GraphNotLoadedError, SplitNotFoundError

if TYPE_CHECKING:
    from hakken_ml_toolkit.ml_base_structures import KnowledgeGraph

    from datasets.common.constants import DataSplits
    from datasets.common.domain import LongTensor2D


class DataLoaderConfig(BaseModel):
    batch_size: int = 128
    num_workers: int = 4


class DataLoaderManager:
    def __init__(
        self,
        config: DataLoaderConfig,
        kg: KnowledgeGraph | None = None,
    ):
        self._initialize(config, kg)

    @staticmethod
    def config_file_path(path: str | Path) -> Path:
        return Path(path) / "config.json"

    def _initialize(
        self,
        config: DataLoaderConfig,
        kg: KnowledgeGraph | None = None,
    ):
        self.kg = kg
        self.config = config

    def set_kg(self, kg: KnowledgeGraph):
        self.kg = kg

    def get_split(self, split: DataSplits) -> LongTensor2D:
        if self.kg is None:
            raise GraphNotLoadedError()

        split_value = split.value
        if split_value == "train":
            return self.kg.facts_dict[split_value]
        if split_value in ["valid", "val"]:
            return self.kg.facts_dict["val"]
        if split_value in ["test"]:
            return self.kg.facts_dict["test"]

        msg = f"Split {split_value} not found in knowledge graph."
        raise SplitNotFoundError(msg)

    def get_data_loader(self, split: DataSplits, **kwargs) -> DataLoader:
        sro_batch = self.get_split(split)

        loader_config = self.config.model_dump()
        if kwargs:
            loader_config.update(kwargs)

        dataset = TensorDataset(sro_batch.data)

        return DataLoader(dataset, **loader_config)

    def get_data_loader_from_dataset(self, dataset: Dataset, **kwargs) -> DataLoader:
        loader_config = self.config.model_dump()
        if kwargs:
            loader_config.update(kwargs)

        return DataLoader(dataset, **loader_config)

    def save_config(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

        config_file_path = DataLoaderManager.config_file_path(path)
        with open(config_file_path, "w") as f:
            json.dump(self.config.model_dump_json(), f)

    @staticmethod
    def load_config(path: str | Path) -> DataLoaderConfig:
        config_file_path = DataLoaderManager.config_file_path(Path(path))

        with open(config_file_path) as f:
            data = json.load(f)

        return DataLoaderConfig.model_validate(data)

    @classmethod
    def load(cls, path: str) -> DataLoaderManager:
        config = cls.load_config(Path(path))
        return cls(config)

    def self_load(self, path: str | Path) -> None:
        config = self.load_config(Path(path))

        self._initialize(config)

    def save(self, path: str | Path):
        self.save_config(Path(path))
