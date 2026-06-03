from __future__ import annotations

import json
from typing import TYPE_CHECKING, Literal

import torch

from hakken_ml_toolkit.ml_base_structures.common.exceptions import (
    InvalidFormatError,
    InvalidTriplesDictKeyError,
    MappingNotFoundError,
    SplitNotInTriplesError,
    TripleNotFoundError,
)
from hakken_ml_toolkit.ml_base_structures.mapping import Mapping
from hakken_ml_toolkit.ml_utils.extras import FactBatchUtils, TensorCreator

if TYPE_CHECKING:
    from pathlib import Path

    from hakken_ml_toolkit.ml_base_structures.common.entities import LongTensor2D
    from hakken_ml_toolkit.ml_base_structures.fact import Fact, FactIndex


class KnowledgeGraph:
    """
    Knowledge graph representation with entity, relation, and optional
    temporal/domain mappings.
    """

    def __init__(  # noqa: PLR0912, PLR0913
        self,
        facts_dict: dict[str, LongTensor2D],
        num_entities: int | None = None,
        num_relations: int | None = None,
        num_timestamps: int | None = None,
        num_domains: int | None = None,
        entity_mapping: Mapping | None = None,
        relation_mapping: Mapping | None = None,
        timestamp_mapping: Mapping | None = None,
        domain_mapping: Mapping | None = None,
        entity_to_domain: dict[int, int] | None = None,
    ) -> None:
        """
        Initialize a KnowledgeGraph.

        Args:
            facts_dict: Dictionary mapping split names to fact batch tensors
            num_entities: Total number of entities (auto-computed if None)
            num_relations: Total number of relations (auto-computed if None)
            num_timestamps: Total number of timestamps (auto-computed if None)
            num_domains: Total number of domains
            entity_mapping: Mapping for entity IDs to indices
            relation_mapping: Mapping for relation IDs to indices
            timestamp_mapping: Mapping for timestamp IDs to indices (for temporal graphs)
            domain_mapping: Mapping for domain IDs to indices
            entity_to_domain: Mapping from entity indices to domain indices
        """
        # Validate facts_dict
        valid_keys = ["all", "train", "val", "test"]
        for key, fact_batch in facts_dict.items():
            if not FactBatchUtils.is_valid(fact_batch):
                msg = f"Invalid fact batch for key '{key}'"
                raise TypeError(msg)

            if key not in valid_keys:
                raise InvalidTriplesDictKeyError(key, valid_keys)

        self.facts_dict = facts_dict

        # Determine temporal nature from first fact batch
        fact_batch = next(iter(facts_dict.values()))
        self._is_temporal = fact_batch.size(1) > 3

        # Set or compute num_entities
        if num_entities is None:
            self.num_entities = fact_batch[:, [0, 2]].unique().numel()
        else:
            self.num_entities = num_entities

        # Set or compute num_relations
        if num_relations is None:
            self.num_relations = fact_batch[:, 1].unique().numel()
        else:
            self.num_relations = num_relations

        # Set or compute num_timestamps (for temporal graphs)
        if num_timestamps is None and self._is_temporal:
            self.num_timestamps = fact_batch[:, 3].unique().numel()
        else:
            self.num_timestamps = num_timestamps

        # Set num_domains
        self.num_domains = num_domains

        # Initialize or set entity mapping
        if entity_mapping is None:
            self.entity_mapping = Mapping.identity(num_elements=self.num_entities)
        else:
            self.entity_mapping = entity_mapping

        # Initialize or set relation mapping
        if relation_mapping is None:
            self.relation_mapping = Mapping.identity(num_elements=self.num_relations)
        else:
            self.relation_mapping = relation_mapping

        # Initialize or set timestamp mapping (for temporal graphs)
        self.timestamp_mapping: Mapping | None
        if timestamp_mapping is None and self._is_temporal:
            self.timestamp_mapping = Mapping.identity(num_elements=self.num_timestamps)
        else:
            self.timestamp_mapping = timestamp_mapping

        # Set domain mapping and entity_to_domain
        self.domain_mapping = domain_mapping
        self.entity_to_domain = entity_to_domain

    def to_device(self, device: str | torch.device) -> None:
        """Move all fact batches to the specified device."""
        for _key, fact_batch in self.facts_dict.items():
            fact_batch.to(device)

    @classmethod
    def load(cls, path: Path) -> KnowledgeGraph:
        """
        Load a KnowledgeGraph from disk.

        Args:
            path: Directory path containing the saved knowledge graph files

        Returns:
            Loaded KnowledgeGraph instance
        """
        with open(path / "data.json") as json_file:
            my_dict: dict = json.load(json_file)

        facts_dict: dict[str, LongTensor2D] = {}
        for key in ["all", "train", "val", "test"]:
            facts_file_path = path / f"facts_{key}.pt"
            if facts_file_path.exists():
                data: torch.Tensor = torch.load(facts_file_path, weights_only=False)

                if data.ndim == 2:
                    facts_dict[key] = data
                else:
                    raise InvalidFormatError()

        entity_mapping = Mapping.load(path / "entity_mapping")
        relation_mapping = Mapping.load(path / "relation_mapping")

        try:
            timestamp_mapping = Mapping.load(path / "timestamp_mapping")
        except MappingNotFoundError:
            timestamp_mapping = None

        num_entities: int | None = my_dict.get("num_entities")
        num_relations: int | None = my_dict.get("num_relations")
        num_timestamps: int | None = my_dict.get("num_timestamps")

        try:
            domain_mapping = Mapping.load(path / "domain_mapping")
            # Load entity-to-domain mapping
            with open(path / "entity_to_domain.json") as f:
                content_items = json.load(f).items()
                entity_to_domain = {int(k): v for k, v in content_items}
        except (MappingNotFoundError, FileNotFoundError):
            domain_mapping = None
            entity_to_domain = None

        num_domains = my_dict.get("num_domains")

        return cls(
            facts_dict=facts_dict,
            num_entities=num_entities,
            num_relations=num_relations,
            num_timestamps=num_timestamps,
            entity_mapping=entity_mapping,
            relation_mapping=relation_mapping,
            timestamp_mapping=timestamp_mapping,
            domain_mapping=domain_mapping,
            entity_to_domain=entity_to_domain,
            num_domains=num_domains,
        )

    def is_temporal(self) -> bool:
        """Check if this knowledge graph includes temporal information."""
        return self._is_temporal

    def save(self, path: Path) -> None:
        """
        Save the KnowledgeGraph to disk.

        Args:
            path: Directory path where the knowledge graph will be saved
        """
        for key, fact_batch in self.facts_dict.items():
            if fact_batch is not None:
                torch.save(fact_batch, path / f"facts_{key}.pt")

        self.entity_mapping.save(path / "entity_mapping")
        self.relation_mapping.save(path / "relation_mapping")

        my_dict = {
            "num_entities": self.num_entities,
            "num_relations": self.num_relations,
        }

        if self.timestamp_mapping is not None:
            self.timestamp_mapping.save(path / "timestamp_mapping")
            my_dict["num_timestamps"] = self.num_timestamps

        if self.domain_mapping is not None:
            self.domain_mapping.save(path / "domain_mapping")
            my_dict["num_domains"] = self.num_domains

            with open(path / "entity_to_domain.json", "w") as f:
                json.dump(self.entity_to_domain, f, indent=4)

        with open(path / "data.json", "w") as json_file:
            json.dump(my_dict, json_file, indent=4)

    def remove_fact_batch(self, split: str, fact_batch: LongTensor2D) -> None:
        """
        Remove a batch of facts from a specific split.

        Args:
            split: Name of the split to remove facts from
            fact_batch: Tensor of facts to remove
        """
        if split not in self.facts_dict:
            raise SplitNotInTriplesError(split=split, valid_splits=list(self.facts_dict.keys()))

        self.facts_dict[split] = FactBatchUtils.remove_batch(
            self.facts_dict[split], fact_batch_to_remove=fact_batch
        )

    def get_entity_indices(self) -> list[int]:
        return self.entity_mapping.get_indexes()

    def get_relation_indices(self) -> list[int]:
        return self.relation_mapping.get_indexes()

    def encode_entity(self, entity_id: str) -> int:
        return self.entity_mapping.encode(entity_id)

    def decode_entity(self, entity_index: int) -> str:
        return self.entity_mapping.decode(entity_index)

    def encode_relation(self, relation_id: str) -> int:
        return self.relation_mapping.encode(relation_id)

    def decode_relation(self, relation_index: int) -> str:
        return self.relation_mapping.decode(relation_index)

    def encode_relations(self, relation_ids: list[str]) -> list[int]:
        return [self.relation_mapping.encode(r) for r in relation_ids]

    def encode_facts(
        self,
        triples_list: list[Fact],
        on_missing: Literal["raise", "ignore"] = "raise",
    ) -> list[FactIndex]:
        """
        Convert a list of triples to tensor indices.

        Args:
            triples_list: List of (subject, relation, object) string tuples
            on_missing: How to handle missing entities/relations ('raise' or 'ignore')

        Returns:
            List of (subject_idx, relation_idx, object_idx) integer tuples
        """
        entity_dict = self.entity_mapping.id_to_index
        relation_dict = self.relation_mapping.id_to_index

        try:
            data_list = [
                (entity_dict[s], relation_dict[r], entity_dict[o]) for (s, r, o) in triples_list
            ]
        except KeyError as e:
            if on_missing == "raise":
                raise TripleNotFoundError() from e
            data_list = [
                (
                    entity_dict.get(s, -1),
                    relation_dict.get(r, -1),
                    entity_dict.get(o, -1),
                )
                for s, r, o in triples_list
            ]

        return data_list

    def encode_facts_as_tensor(
        self,
        triples_list: list[tuple[str, str, str]],
        on_missing: Literal["raise", "ignore"] = "raise",
    ) -> torch.Tensor:
        """
        Convert a list of triples to a tensor of indices.

        Args:
            triples_list: List of (subject, relation, object) string tuples
            on_missing: How to handle missing entities/relations ('raise' or 'ignore')

        Returns:
            Tensor of shape (num_triples, 3) with integer indices
        """
        data_list = self.encode_facts(triples_list=triples_list, on_missing=on_missing)
        return TensorCreator.long_tensor(data_list)

    def decode_facts(self, fact_indices: list[FactIndex] | torch.Tensor) -> list[Fact]:
        """
        Convert FactIndex objects back to Fact objects.

        Args:
            fact_indices: List of FactIndex objects or tensor of fact indices

        Returns:
            List of Fact objects with string identifiers
        """
        if isinstance(fact_indices, torch.Tensor):
            fact_indices = fact_indices.tolist()

        return [self.decode_fact(fact_index) for fact_index in fact_indices]

    def decode_fact(self, triple_index: FactIndex) -> Fact:
        """
        Convert a single TripleIndex to a Triple with string identifiers.

        Args:
            triple_index: TripleIndex with integer indices

        Returns:
            Triple with string identifiers
        """
        subject_index = triple_index[0]
        relation_index = triple_index[1]
        object_index = triple_index[2]
        subject = self.entity_mapping.index_to_id[subject_index]
        relation = None
        if relation_index is not None:
            relation = self.relation_mapping.index_to_id[relation_index]
        object = self.entity_mapping.index_to_id[object_index]

        return (subject, relation, object)

    def get_encoded_facts(self, splits: list[str] | None = None) -> LongTensor2D:
        if splits is None:
            splits = list(self.facts_dict.keys())

        return torch.cat(
            [self.facts_dict[split_name] for split_name in splits],
            dim=0,
        )
