from __future__ import annotations

from functools import cached_property
from typing import Literal, cast

import torch
from datasets.common.constants import DataSplits
from hakken_ml_toolkit.ml_base_structures import KnowledgeGraph
from hakken_ml_toolkit.ml_utils.extras import FactBatchUtils
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from kge.common.entities import KGData
from kge.common.exceptions import (
    GraphNotLoadedError,
    NotInitializedError,
    SplitsError,
)
from kge.common.types import LongTensor2D
from kge.data_loaders.mimic_kge import MimicKGEDataLoader, MimicKGEDataLoaderConfig
from kge.data_processor.config import KGDataProcessorConfig
from kge.models.kge_api import KGEAPI

Data_Processing_Token = "data_processing"

DEFAULT_CONFIG = KGDataProcessorConfig()


class KGDataProcessor:
    def __init__(
        self,
        config: KGDataProcessorConfig = DEFAULT_CONFIG,
        kg: KnowledgeGraph | None = None,
    ):
        self._initialize(config, kg)

    @cached_property
    def loader_config(self) -> DictConfig:
        return cast("DictConfig", OmegaConf.create(self.config.loader))

    def _initialize(
        self,
        config: KGDataProcessorConfig,
        kg: KnowledgeGraph | None = None,
    ):
        self._kg = kg

        self.config = config

    def set_kg(self, kg: KnowledgeGraph):
        self._kg = kg

    @property
    def kg(self) -> KnowledgeGraph:
        if self._kg is None:
            raise GraphNotLoadedError()

        return self._kg

    def relation_list(self) -> list[str]:
        return self.kg.relation_mapping.get_ids()

    def get_split(self, split: DataSplits) -> LongTensor2D:
        split_value = split.value

        if split_value == "valid":
            split_value = "val"

        if split_value not in self.kg.facts_dict:
            msg = f"Split {split_value} not found in knowledge graph {self.kg.facts_dict.keys()}"
            raise SplitsError(msg)
        return self.kg.facts_dict[split_value]

    def to(self, device: str | torch.device) -> None:
        """
        Moves all device-dependent components to the specified device.

        Args:
            device: Target device as string ('cuda:0', 'cpu') or torch.device object
        """
        pass

    def get_sro_batch(
        self,
        triples_list: list[tuple[str, str, str]],
        on_missing: Literal["raise", "ignore"] = "raise",
    ) -> LongTensor2D:
        if self._kg is None:
            msg = "Knowledge graph is not set"

            raise NotInitializedError(msg)

        return self.kg.encode_facts_as_tensor(triples_list=triples_list, on_missing=on_missing)

    def get_supervised_dataset(
        self, split: DataSplits, num_triples: int | None = None
    ) -> TensorDataset:
        sro_batch = self.get_split(split)
        if num_triples is not None:
            sro_batch = sro_batch[:num_triples]

        so_batch, target_batch = FactBatchUtils.to_so_batch_and_relations(
            sro_batch, num_relations=self.kg.num_relations
        )
        return TensorDataset(so_batch, target_batch)

    def get_dataset(self, split: DataSplits, num_triples: int | None = None) -> TensorDataset:
        sro_batch = self.get_split(split)
        if num_triples is not None:
            sro_batch = sro_batch[:num_triples]
        return TensorDataset(sro_batch)

    def get_data_loader(
        self, split: DataSplits, num_triples: int | None = None, **kwargs
    ) -> DataLoader:
        dataset = self.get_dataset(split=split, num_triples=num_triples)

        loader_config = self.config.loader.copy()
        if kwargs:
            loader_config.update(kwargs)

        return DataLoader(dataset, **loader_config)

    def get_mimic_kge_data_loader(
        self,
        split: DataSplits,
        trained_kge: KGEAPI,
        subgraph_split: list[DataSplits] | DataSplits | None = None,
        **kwargs,
    ) -> MimicKGEDataLoader:
        facts = self.get_split(split)
        data = KGData.from_facts(
            facts,
            num_nodes=self.kg.num_entities,
            num_relations=self.kg.num_relations,
        )
        loader_dict = self.config.loader.copy()
        loader_dict.update(kwargs)
        loader_dict["num_relations"] = data.num_relations

        subgraph_data = data

        subgraph_facts: LongTensor2D | None = None
        if isinstance(subgraph_split, list) and len(subgraph_split) > 0:
            subgraph_facts_list = []
            for split_i in subgraph_split:
                subgraph_facts_list.append(self.get_split(split_i))
            subgraph_facts = torch.cat(subgraph_facts_list, dim=0)
        elif isinstance(subgraph_split, DataSplits):
            subgraph_facts = self.get_split(subgraph_split)

        if subgraph_facts is not None:
            subgraph_data = KGData.from_facts(
                subgraph_facts,
                num_nodes=self.kg.num_entities,
                num_relations=self.kg.num_relations,
            )

        loader_config = MimicKGEDataLoaderConfig.model_validate(loader_dict)

        return MimicKGEDataLoader.from_config(
            config=loader_config,
            data=data,
            trained_kge=trained_kge,
            subgraph_data=subgraph_data,
        )
