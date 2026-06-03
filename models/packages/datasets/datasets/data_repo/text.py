from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pandas as pd
from hakken_ml_toolkit.ml_base_structures import KnowledgeGraph, Mapping
from hakken_ml_toolkit.ml_utils import DSVUtils
from hakken_ml_toolkit.ml_utils.extras import TensorCreator
from loguru import logger

from datasets.common.exceptions import InvalidDateTypeError
from datasets.data_repo.base import DataRepositoryConfig, DataRepositoryI

if TYPE_CHECKING:
    from datasets.common.domain import LongTensor2D


class TextKGConfig(DataRepositoryConfig):
    files_dict: dict[str, str]
    relation_mapping: dict[str, Any] | None = None
    entity_mapping: dict[str, Any] | None = None
    delimiter: str = "\t"
    column_names: list[str] | None = None
    column_rename: dict[str, str] | None = None
    date_format: str | None = None


class TextKGDataset(DataRepositoryI[TextKGConfig]):
    def __init__(self, config: TextKGConfig):
        super().__init__(config)

        self.subject_column = "subject"

        self.relation_column = "relation"

        self.object_column = "object"

        self.date_column = "date"

        self.kg = None

        self.is_temporal = (
            self.config.column_names is not None and len(self.config.column_names) == 4
        )

    @classmethod
    def _get_config_class(cls) -> type[TextKGConfig]:
        return TextKGConfig

    def load_mapping(
        self,
        file_path: str,
        delimiter: str,
        column_names: list[str] | None = None,
        column_rename: dict[str, str] | None = None,
    ) -> Mapping:
        df = DSVUtils.read_dsv(file_path=file_path, delimiter=delimiter, names=column_names)

        if column_rename is not None:
            df = df.rename(columns=column_rename)

        id_to_index = cast(
            "dict[str, int]", df.set_index(df["ids"].astype(str))["indexes"].astype(int).to_dict()
        )
        index_to_id = cast(
            "dict[int, str]", df.set_index(df["indexes"].astype(int))["ids"].astype(str).to_dict()
        )

        return Mapping(id_to_index=id_to_index, index_to_id=index_to_id)

    def validate_entity_mapping(self, entity_mapping: Mapping, edges_df: pd.DataFrame) -> None:
        entity_list = set(entity_mapping.id_to_index.keys())

        subjects = set(edges_df[self.subject_column].unique())
        objects = set(edges_df[self.object_column].unique())

        all_entities = subjects.union(objects)

        if entity_list != all_entities:
            extra_in_mapping = entity_list - all_entities
            missing_in_mapping = all_entities - entity_list
            error_message = []
            if extra_in_mapping:
                logger.warning(f"Entities in mapping but not in edges: {extra_in_mapping}")
            if missing_in_mapping:
                error_message.append(f"Entities in edges but not in mapping: {missing_in_mapping}")

                raise ValueError("\n".join(error_message))

    def validate_relation_mapping(self, relation_mapping: Mapping, edges_df: pd.DataFrame) -> None:
        relation_list = set(relation_mapping.id_to_index.keys())

        all_relations = set(edges_df[self.relation_column].unique())

        if relation_list != all_relations:
            extra_in_mapping = relation_list - all_relations
            missing_in_mapping = all_relations - relation_list
            error_message = []
            if extra_in_mapping:
                logger.warning(f"Relations in mapping but not in edges: {extra_in_mapping}")
            if missing_in_mapping:
                error_message.append(f"Relations in edges but not in mapping: {missing_in_mapping}")

                raise ValueError("\n".join(error_message))

    def _load_from_database(self) -> KnowledgeGraph:
        facts_df = {}

        root_folder = Path(self.config.root_folder)
        if not root_folder.exists():
            msg = f"Root folder '{self.config.root_folder}' does not exist."
            raise FileNotFoundError(msg)
        for key, file_name in self.config.files_dict.items():
            file_path = root_folder / file_name
            df_i = DSVUtils.read_dsv(
                file_path=file_path,
                delimiter=self.config.delimiter,
                names=self.config.column_names,
            )

            if self.config.column_rename is not None:
                raise NotImplementedError()

            facts_df[key] = df_i

        edges_df = pd.concat(list(facts_df.values()))

        if self.config.entity_mapping is not None:
            entity_mapping = self.load_mapping(
                file_path=str(root_folder / self.config.entity_mapping["filename"]),
                delimiter=self.config.entity_mapping["delimiter"],
                column_names=self.config.entity_mapping.get("column_names", None),
                column_rename=self.config.entity_mapping.get("column_rename", None),
            )

            self.validate_entity_mapping(entity_mapping=entity_mapping, edges_df=edges_df)

        else:
            entity_mapping = self._create_entity_mapping(edges_df)
        if self.config.relation_mapping is not None:
            relation_mapping = self.load_mapping(
                file_path=str(root_folder / self.config.relation_mapping["filename"]),
                delimiter=self.config.relation_mapping["delimiter"],
                column_names=self.config.relation_mapping.get("column_names", None),
                column_rename=self.config.relation_mapping.get("column_rename", None),
            )
            self.validate_relation_mapping(relation_mapping=relation_mapping, edges_df=edges_df)
        else:
            relation_mapping = self._create_relation_mapping(edges_df)

        timestamp_mapping = None
        if self.is_temporal:
            timestamp_mapping = self._create_timestamp_mapping(edges_df)

        facts_dict = {}
        for key, df_i in facts_df.items():
            facts_i = self._create_facts(
                df_i, entity_mapping, relation_mapping, timestamp_map=timestamp_mapping
            )
            facts_dict[key] = facts_i

        num_timestamps = None
        if self.is_temporal and timestamp_mapping is not None:
            num_timestamps = len(timestamp_mapping.index_to_id)

        return KnowledgeGraph(
            facts_dict=facts_dict,
            num_entities=len(entity_mapping.id_to_index),
            num_relations=len(relation_mapping.id_to_index),
            num_timestamps=num_timestamps,
            entity_mapping=entity_mapping,
            relation_mapping=relation_mapping,
            timestamp_mapping=timestamp_mapping,  # type: ignore
        )

    def _create_entity_mapping(self, edges_df: pd.DataFrame) -> Mapping:
        """
        Create entity ID to index mapping from edges dataframe.

        Args:
            edges_df: DataFrame with 'subject' and 'object' columns

        Returns:
            Mapping object with id_to_index and index_to_id dictionaries
        """
        ocid_subject = set(edges_df["subject"].unique())
        ocid_object = set(edges_df["object"].unique())
        ocid_nodes = sorted(ocid_subject.union(ocid_object))

        id_to_index = {ocid: i for i, ocid in enumerate(ocid_nodes)}
        index_to_id = dict(enumerate(ocid_nodes))

        return Mapping(id_to_index=id_to_index, index_to_id=index_to_id)

    def _create_relation_mapping(self, edges_df: pd.DataFrame) -> Mapping:
        """
        Create relation ID to index mapping from edges dataframe.

        Args:
            edges_df: DataFrame with 'relation' column

        Returns:
            Mapping object with id_to_index and index_to_id dictionaries
        """
        relations = sorted(edges_df["relation"].unique())

        id_to_index = {ocid: i for i, ocid in enumerate(relations)}

        index_to_id = dict(enumerate(relations))
        return Mapping(id_to_index=id_to_index, index_to_id=index_to_id)

    def _create_timestamp_mapping(self, edges_df: pd.DataFrame) -> Mapping:
        """
        Create timestamp to index mapping from edges dataframe.

        Args:
            edges_df: DataFrame with date column

        Returns:
            Mapping object for timestamps

        Raises:
            InvalidDateTypeError: If date column is not integer and no date_format provided
        """
        dtype = edges_df[self.date_column].dtype

        if self.config.date_format is not None:
            edges_df["timestamp"] = pd.to_datetime(
                edges_df[self.date_column], format=self.config.date_format
            )

            df_sorted = edges_df.sort_values("timestamp")
            sorted_timestamps = df_sorted[self.date_column].unique().tolist()
            edges_df = edges_df.drop("timestamp", axis=1)

        else:
            if dtype != "int64":
                raise InvalidDateTypeError()

            sorted_timestamps = sorted(edges_df[self.date_column].unique())

        id_to_index = {ts: i for i, ts in enumerate(sorted_timestamps)}
        index_to_id = dict(enumerate(sorted_timestamps))

        return Mapping(id_to_index=id_to_index, index_to_id=index_to_id)

    def _create_facts(
        self,
        edges_df: pd.DataFrame,
        entity_map: Mapping,
        relation_map: Mapping,
        timestamp_map: Mapping | None = None,
    ) -> LongTensor2D:
        """
        Convert edges dataframe to tensor of fact indices.

        Args:
            edges_df: DataFrame with edge data
            entity_map: Entity ID to index mapping
            relation_map: Relation ID to index mapping
            timestamp_map: Optional timestamp mapping for temporal KGs

        Returns:
            Tensor of fact indices (shape: num_facts * 3 or num_facts * 4)
        """

        subject_id = edges_df[self.subject_column].map(entity_map.id_to_index).astype(int)
        relation_id = edges_df[self.relation_column].map(relation_map.id_to_index).astype(int)
        object_id = edges_df[self.object_column].map(entity_map.id_to_index).astype(int)

        if timestamp_map is not None:
            timestamp_id = edges_df[self.date_column].map(timestamp_map.id_to_index).astype(int)
            # Create a DataFrame from the IDs
            facts_df = pd.concat([subject_id, relation_id, object_id, timestamp_id], axis=1)
            facts_df.columns = pd.Index(
                [
                    "subject_index",
                    "relation_index",
                    "object_index",
                    "timestamp_index",
                ]
            )

            facts_df = facts_df.sort_values("timestamp_index", ascending=True)

            return TensorCreator.long_tensor(facts_df.values)
        facts_np = pd.concat([subject_id, relation_id, object_id], axis=1).values

        return TensorCreator.long_tensor(facts_np)
