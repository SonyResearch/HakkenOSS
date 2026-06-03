import pandas as pd
import torch
from hakken_ml_toolkit.ml_base_structures import Mapping
from loguru import logger

from datasets.common.domain import LongTensor2D


class DataFrameUtils:
    @staticmethod
    def create_id_index_mapping(facts_df: pd.DataFrame, id_columns: list[str]) -> Mapping:
        unique_ids: set[str] = set()
        for column in id_columns:
            if column in facts_df.columns:
                unique_ids.update(facts_df[column].unique())

        all_ids = sorted(unique_ids)

        id_to_index: dict[str, int] = {}
        index_to_id: dict[int, str] = {}
        for index, string_id in enumerate(all_ids):
            id_to_index[string_id] = index
            index_to_id[index] = string_id

        return Mapping(id_to_index=id_to_index, index_to_id=index_to_id)

    @staticmethod
    def create_mappings_from_df(
        facts_df: pd.DataFrame,
        subject_column: str = "subject_id",
        relation_column: str = "relation_id",
        object_column: str = "object_id",
        timestamp_column: str | None = None,
    ) -> dict[str, Mapping]:
        logger.info(f"Creating knowledge graph tensors from {len(facts_df)} facts")
        entity_mapping = DataFrameUtils.create_id_index_mapping(
            facts_df=facts_df, id_columns=[subject_column, object_column]
        )

        relation_mapping = DataFrameUtils.create_id_index_mapping(
            facts_df=facts_df, id_columns=[relation_column]
        )

        timestamp_mapping = None
        if timestamp_column is not None:
            timestamp_mapping = DataFrameUtils.create_id_index_mapping(
                facts_df=facts_df, id_columns=[timestamp_column]
            )

        mapping_dict = {}

        mapping_dict["entity"] = entity_mapping
        mapping_dict["relation"] = relation_mapping
        if timestamp_column is not None and timestamp_mapping is not None:
            mapping_dict["timestamp"] = timestamp_mapping

        return mapping_dict

    @staticmethod
    def build_entity_domain_df(
        facts_df: pd.DataFrame,
        entity_mapping: Mapping,
        subject_column: str = "subject_id",
        object_column: str = "object_id",
        subject_domain_column: str = "subject_domain",
        object_domain_column: str = "object_domain",
    ) -> tuple[Mapping, pd.DataFrame]:
        """
        Build a per-entity DataFrame with entity/domain IDs and their corresponding indices.
        Also, it creates a domain Mapping from the union of the subject and object domain columns.

        Args:
            facts_df: DataFrame containing at least the entity and domain columns.
            entity_mapping: Mapping for entity IDs -> indices (shared for subjects/objects).
            subject_column: Column name for subject entity IDs (default: "subject_id").
            object_column: Column name for object entity IDs (default: "object_id").
            subject_domain_column: Column name for subject domain IDs (default: "subject_domain").
            object_domain_column: Column name for object domain IDs (default: "object_domain").

        Returns:
            A tuple of:
            - domain_mapping: Mapping with id_to_index / index_to_id for domain IDs.
            - entities_df:   DataFrame with columns
                            ["entity_id", "entity_index", "domain_id", "domain_index"].

        Notes:
            - If an entity appears with multiple domains, the first seen (row order) is kept.
            Change the deduplication policy if you need e.g. most frequent domain.
            - The function assumes that all entity IDs present in the DataFrame exist in
            `entity_mapping`.
        """

        domain_mapping = DataFrameUtils.create_id_index_mapping(
            facts_df=facts_df,
            id_columns=[subject_domain_column, object_domain_column],
        )
        subject_entities = (
            facts_df[[subject_column, subject_domain_column]]
            .rename(
                columns={
                    subject_column: "entity_id",
                    subject_domain_column: "domain_id",
                }
            )
            .drop_duplicates()
        )
        object_entities = (
            facts_df[[object_column, object_domain_column]]
            .rename(columns={object_column: "entity_id", object_domain_column: "domain_id"})
            .drop_duplicates()
        )

        entities_df: pd.DataFrame = pd.concat(
            [subject_entities, object_entities], ignore_index=True
        )

        entities_df = entities_df.drop_duplicates(subset=["entity_id"], keep="first")

        entities_df["entity_index"] = (
            entities_df["entity_id"].map(entity_mapping.id_to_index).astype(int)
        )
        entities_df["domain_index"] = (
            entities_df["domain_id"].map(domain_mapping.id_to_index).astype(int)
        )

        entities_df = (
            entities_df[["entity_id", "entity_index", "domain_id", "domain_index"]]
            .sort_values("entity_index")
            .reset_index(drop=True)
        )

        return domain_mapping, entities_df

    # ruff: noqa: PLR0913
    @staticmethod
    def create_torch_facts(
        facts_df: pd.DataFrame,
        entity_map: Mapping,
        relation_map: Mapping,
        subject_column: str = "subject_id",
        relation_column: str = "relation_id",
        object_column: str = "object_id",
        timestamp_column: str | None = None,
        timestamp_map: Mapping | None = None,
    ) -> LongTensor2D:
        """Convert DataFrame of facts to PyTorch tensor with mapped indices.

        Maps string IDs to numerical indices using provided mappings and returns
        a tensor.

        Args:
            facts_df: DataFrame containing fact triples
            entity_map: Mapping from entity IDs to numerical indices
            relation_map: Mapping from relation IDs to numerical indices
            subject_column: Column name for subject entities
            relation_column: Column name for relations
            object_column: Column name for object entities
            timestamp_column: Optional column name for timestamps
            timestamp_map: Optional mapping for timestamp IDs to indices

        Returns:
            LongTensor with shape (n_facts, 3) or (n_facts, 4) if timestamps included
        """
        ids_list = []

        subject_id = facts_df[subject_column].map(entity_map.id_to_index).astype(int)
        relation_id = facts_df[relation_column].map(relation_map.id_to_index).astype(int)
        object_id = facts_df[object_column].map(entity_map.id_to_index).astype(int)

        ids_list = [subject_id, relation_id, object_id]
        if timestamp_column is not None and timestamp_map is not None:
            timestamp_id = facts_df[timestamp_column].map(timestamp_map.id_to_index).astype(int)
            ids_list.append(timestamp_id)

        triples_np = pd.concat(ids_list, axis=1).values

        return torch.LongTensor(triples_np)
