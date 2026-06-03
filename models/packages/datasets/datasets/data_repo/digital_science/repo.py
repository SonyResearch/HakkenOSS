from __future__ import annotations

# ruff: noqa: TC003
from pathlib import Path

from hakken_ml_toolkit.ml_base_structures import KnowledgeGraph
from hakken_ml_toolkit.ml_utils.extras import PyTorchUtils
from omegaconf import MISSING

from datasets.data_repo.base import DataRepositoryConfig, DataRepositoryI
from datasets.data_repo.digital_science.utils import DigitalScienceUtils
from datasets.utils import DataFrameUtils


class DigitalScienceConfig(DataRepositoryConfig):
    nodes_file: str | Path = MISSING
    edges_file: str | Path = MISSING
    relation_column: str = "relation"
    subject_column: str = "ocid_subject"
    object_column: str = "ocid_object"
    timestamp_column: str | None = None


class DigitalScience(DataRepositoryI[DigitalScienceConfig]):
    name = "digital_science"

    def __init__(self, config: DigitalScienceConfig):
        super().__init__(config)

    @classmethod
    def _get_config_class(cls) -> type[DigitalScienceConfig]:
        return DigitalScienceConfig

    def _load_from_database(self) -> KnowledgeGraph:
        facts_df = DigitalScienceUtils.load_filtered_edges_df(
            edges_file=self.config.edges_file,
            relation_filter=self.config.relation_filter,
            relation_column=self.config.relation_column,
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

        facts_dict = PyTorchUtils.split_tensors(
            facts_tensor, split_proportion=self.config.data_split_proportion_dict
        )

        entity_mapping = mapping_dict["entity"]
        relation_mapping = mapping_dict["relation"]
        timestamp_mapping = mapping_dict.get("timestamps", None)

        num_timestamps = None
        if timestamp_mapping is not None:
            num_timestamps = len(timestamp_mapping.id_to_index)

        return KnowledgeGraph(
            facts_dict=facts_dict,
            num_entities=len(entity_mapping.id_to_index),
            num_relations=len(relation_mapping.id_to_index),
            num_timestamps=num_timestamps,
            entity_mapping=entity_mapping,
            relation_mapping=relation_mapping,
            timestamp_mapping=timestamp_mapping,  # type: ignore
        )
