from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, cast

import pandas as pd
from hakken_ml_toolkit.ml_base_structures import KnowledgeGraph, Mapping
from hakken_ml_toolkit.ml_utils import DSVUtils
from hakken_ml_toolkit.ml_utils.extras import PyTorchUtils
from loguru import logger

from datasets.data_repo.base import DataRepositoryI
from datasets.data_repo.dsv.config import DSVKGConfig
from datasets.utils import DataFrameUtils

if TYPE_CHECKING:
    from torch import Tensor


class DSVKGDataset(DataRepositoryI[DSVKGConfig]):
    name = "dsv"

    def __init__(self, config: DSVKGConfig):
        super().__init__(config)

    @classmethod
    def _get_config_class(cls) -> type[DSVKGConfig]:
        return DSVKGConfig

    def _temporal_split(
        self, facts_tensor: Tensor, timestamp_mapping: Mapping
    ) -> dict[str, Tensor]:
        if facts_tensor.shape[1] != 4:
            msg = "Expected facts_tensor with 4 columns"
            raise RuntimeError(msg)

        if self.config.data_split_temporal_dict is None:
            msg = "Temporal split configuration (data_split_temporal_dict) is missing."
            raise RuntimeError(msg)
        facts_dict = {}

        timestamp = facts_tensor[:, 3]

        for split_name, date_range in self.config.data_split_temporal_dict.items():
            min_date, max_date = date_range
            min_index = timestamp_mapping.id_to_index[cast("str", min_date)]
            max_index = timestamp_mapping.id_to_index[cast("str", max_date)]
            cond = (timestamp >= min_index) & (timestamp <= max_index)

            facts_dict[split_name] = facts_tensor[cond, :3]

        return facts_dict

    def _load_from_database(self) -> KnowledgeGraph:
        dtype = None
        if self.config.facts_file_columns_dtypes is not None:
            dtype = defaultdict(lambda: "string", self.config.facts_file_columns_dtypes)

        all_facts_df = DSVUtils.read_dsv(
            self.config.facts_file,
            delimiter=self.config.facts_file_delimiter,
            header=0,
            dtype=dtype,
        )

        if self.config.timestamp_parser:
            all_facts_df[self.config.timestamp_column] = pd.to_datetime(
                all_facts_df["timestamp"],
                errors="coerce",
                **self.config.timestamp_parser,
            )
            all_facts_df = all_facts_df.dropna(subset=[self.config.timestamp_column])

            all_facts_df[self.config.timestamp_column] = all_facts_df[
                self.config.timestamp_column
            ].dt.year

        facts_df = self.prune_invalid_facts(
            all_facts_df,
            time_col=self.config.timestamp_column,
            subject_id_col=self.config.subject_column,
            object_id_col=self.config.object_column,
        )

        mapping_dict = DataFrameUtils.create_mappings_from_df(
            facts_df=facts_df,
            subject_column=self.config.subject_column,
            relation_column=self.config.relation_column,
            object_column=self.config.object_column,
            timestamp_column=self.config.timestamp_column,
        )

        entity_mapping = mapping_dict["entity"]
        relation_mapping = mapping_dict["relation"]
        timestamp_mapping = mapping_dict.get("timestamp", None)

        facts_tensor = DataFrameUtils.create_torch_facts(
            facts_df=facts_df,
            entity_map=entity_mapping,
            relation_map=relation_mapping,
            subject_column=self.config.subject_column,
            relation_column=self.config.relation_column,
            object_column=self.config.object_column,
            timestamp_column=self.config.timestamp_column,
            timestamp_map=timestamp_mapping,
        )

        domain_mapping: Mapping | None = None
        num_domains: int | None = None
        entity_to_domain: dict[int, int] | None = None

        if (
            self.config.subject_domain_column is not None
            and self.config.object_domain_column is not None
        ):
            domain_mapping, entity_domain_df = DataFrameUtils.build_entity_domain_df(
                facts_df=facts_df,
                entity_mapping=entity_mapping,
                subject_column=self.config.subject_column,
                object_column=self.config.object_column,
                subject_domain_column=self.config.subject_domain_column,
                object_domain_column=self.config.object_domain_column,
            )

            entity_to_domain = cast(
                "dict[int, int]",
                entity_domain_df.set_index("entity_index")["domain_index"].to_dict(),
            )

            num_domains = len(domain_mapping)

        if self.config.data_split_proportion_dict is not None:
            logger.info(
                f"Splitting data with proportions: {self.config.data_split_proportion_dict}"
            )

            facts_dict = PyTorchUtils.split_tensors(
                facts_tensor,
                split_proportion=self.config.data_split_proportion_dict,
                shuffle=False,
            )
        elif self.config.data_split_temporal_dict is not None:
            if timestamp_mapping is None:
                msg = "Please set the timestamp column in the config"
                raise RuntimeError(msg)
            facts_dict = self._temporal_split(
                facts_tensor=facts_tensor, timestamp_mapping=timestamp_mapping
            )

        num_timestamps = None
        if timestamp_mapping is not None:
            num_timestamps = len(timestamp_mapping.id_to_index)

        return KnowledgeGraph(
            facts_dict=facts_dict,
            num_entities=len(entity_mapping.id_to_index),
            num_relations=len(relation_mapping.id_to_index),
            num_timestamps=num_timestamps,
            num_domains=num_domains,
            entity_mapping=entity_mapping,
            relation_mapping=relation_mapping,
            timestamp_mapping=timestamp_mapping,  # type: ignore
            domain_mapping=domain_mapping,  # type: ignore
            entity_to_domain=entity_to_domain,
        )
