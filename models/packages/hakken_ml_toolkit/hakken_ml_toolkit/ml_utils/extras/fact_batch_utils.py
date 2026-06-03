from typing import cast

import networkx as nx
import torch
from loguru import logger

from hakken_ml_toolkit.ml_utils.extras.domain import (
    FloatTensor2D,
    LongTensor1D,
    LongTensor2D,
    ProximityNetworkData,
)


class FactBatchUtils:
    @staticmethod
    def is_valid(fact_batch: LongTensor2D) -> bool:
        if fact_batch.ndim != 2 or fact_batch.shape[1] < 3:
            msg = f"Incorrect dimensions. ndim={fact_batch.ndim}, shape={fact_batch.shape}"
            logger.warning(msg)
            return False
        if fact_batch.dtype != torch.long:
            msg = f"Incorrect dtype of fact_batch. Expected torch.long, got {fact_batch.dtype}"
            logger.warning(msg)

            return False
        if fact_batch.numel() > 0 and fact_batch.min() < 0:
            logger.warning(
                f"Invalid fact_batch: contains negative values. Min value: {fact_batch.min()}"
            )
            return False
        if not torch.isfinite(fact_batch).all():
            logger.warning("Invalid fact_batch: contains non-finite values")
            return False
        return True

    @staticmethod
    def num_entities(fact_batch: LongTensor2D) -> int:
        unique_entities: torch.Tensor = fact_batch[:, [0, 2]].unique()
        return int(unique_entities.numel())

    @staticmethod
    def num_relations(fact_batch: LongTensor2D) -> int:
        unique_relations: torch.Tensor = fact_batch[:, 1].unique()
        return int(unique_relations.numel())

    @staticmethod
    def subject(fact_batch: LongTensor2D) -> LongTensor1D:
        return fact_batch[:, 0]

    @staticmethod
    def relation(fact_batch: LongTensor2D) -> LongTensor1D:
        return fact_batch[:, 1]

    @staticmethod
    def object(fact_batch: LongTensor2D) -> LongTensor1D:
        return fact_batch[:, 2]

    @staticmethod
    def timestamp(fact_batch: LongTensor2D) -> LongTensor1D:
        return fact_batch[:, 3]

    @staticmethod
    def entities(fact_batch: LongTensor2D) -> LongTensor1D:
        subjects = fact_batch[:, 0]
        objects = fact_batch[:, 2]
        all_entities = torch.cat((subjects, objects))
        unique_entities = torch.unique(all_entities)
        unique_entities_sorted, _ = torch.sort(unique_entities)
        return cast("LongTensor1D", unique_entities_sorted)

    @staticmethod
    def to_so_batch(fact_batch: LongTensor2D) -> LongTensor2D:
        return fact_batch[:, [0, 2]]

    @staticmethod
    def to_sr_batch(fact_batch: LongTensor2D) -> LongTensor2D:
        return fact_batch[:, [0, 1]]

    @staticmethod
    def to_ro_batch(fact_batch: LongTensor2D) -> LongTensor2D:
        return fact_batch[:, [1, 2]]

    @staticmethod
    def _so_unique_relation_targets(
        fact_batch: LongTensor2D, num_relations: int | None = None
    ) -> tuple[LongTensor2D, LongTensor2D, LongTensor1D]:
        """One ``torch.unique`` pass: unique (s,o), multi-hot per unique pair, inverse per row.

        Uses only the first three columns ``[s, r, o]`` (extra columns ignored).
        """
        sro = fact_batch[:, :3]
        so_batch = FactBatchUtils.to_so_batch(sro)
        so_unique_batch, inverse_indices = torch.unique(so_batch, dim=0, return_inverse=True)
        relations = sro[:, 1]
        if num_relations is None:
            num_relations = int(relations.max().item() + 1)

        so_batch_size = so_unique_batch.shape[0]
        target = torch.zeros((so_batch_size, num_relations), dtype=torch.long)

        for i in range(sro.shape[0]):
            relation = int(sro[i, 1].item())
            unique_index = int(inverse_indices[i].item())
            target[unique_index, relation] = 1

        return so_unique_batch, target, cast("LongTensor1D", inverse_indices)

    @staticmethod
    def to_so_batch_and_relations(
        fact_batch: LongTensor2D, num_relations: int | None = None
    ) -> tuple[LongTensor2D, LongTensor2D]:
        """
        Converts a batch of SRO facts to unique subject-object pairs and one-hot relation
        targets.

        Args:
            fact_batch: Tensor of shape (batch_size, 3) containing [subject, relation, object]
                    triples
            num_relations: Total number of possible relations. If None, inferred from data

        Returns:
            tuple containing:
                - Tensor of unique (subject, object) pairs
                - One-hot encoded relation targets for each unique pair
        """

        so_unique_batch, target, _ = FactBatchUtils._so_unique_relation_targets(
            fact_batch, num_relations=num_relations
        )
        return so_unique_batch, target

    @staticmethod
    def fact_batch_pair_relation_labels(
        fact_batch: LongTensor2D, num_relations: int | None = None
    ) -> FloatTensor2D:
        """Multi-hot relation labels per fact row from (s, o) pair aggregation.

        For each row, builds the same multi-hot vector as :meth:`to_so_batch_and_relations`
        for that row's ``(subject, object)``: every relation that appears on that pair
        anywhere in ``fact_batch`` (uses the first three columns ``[s, r, o]``; any extra
        columns such as timestamps are ignored).

        This expands the unique-pair tensors from :meth:`to_so_batch_and_relations` back
        to one row per input fact, for training objectives that need labels aligned with
        full fact batches (e.g. KGE + BCE on relations).

        Args:
            fact_batch: ``[N, >=3]`` long tensor of facts.
            num_relations: Size of the relation dimension. If ``None``, inferred from data.

        Returns:
            Float tensor ``[N, num_relations]`` (dtype float32), suitable for BCE-style losses.
        """
        if fact_batch.numel() == 0:
            nr = num_relations if num_relations is not None else 0
            return torch.zeros((0, nr), dtype=torch.float32, device=fact_batch.device)

        _, target, inverse_indices = FactBatchUtils._so_unique_relation_targets(
            fact_batch, num_relations=num_relations
        )
        return target[inverse_indices].to(dtype=torch.float32)

    @staticmethod
    def so_to_sro_batch(so_batch: LongTensor2D, num_relations: int) -> LongTensor2D:
        """
        Expands subject-object pairs into all possible subject-relation-object triples.

        For each (subject, object) pair in the input batch, creates num_relations triples
        by combining the pair with every possible relation ID from 0 to num_relations-1.

        Args:
            so_batch: Tensor of shape (batch_size, 2) containing [subject, object] pairs
            num_relations: Total number of relations to generate for each pair

        Returns:
            Tensor of shape (batch_size * num_relations, 3) containing all possible
            [subject, relation, object] triples

        Note:
             This method assumes continguous relation IDs from 0 to num_relations
        """
        batch_size = so_batch.size(0)
        all_relations = torch.arange(num_relations, device=so_batch.device)

        expanded_relations = all_relations.unsqueeze(0).expand(batch_size, -1).flatten()
        expanded_so = so_batch.repeat_interleave(num_relations, dim=0)

        sro_batch = torch.empty(
            (len(expanded_relations), 3), dtype=torch.long, device=so_batch.device
        )
        sro_batch[:, 0] = expanded_so[:, 0]
        sro_batch[:, 1] = expanded_relations
        sro_batch[:, 2] = expanded_so[:, 1]

        return sro_batch

    @staticmethod
    def ro_to_sro_batch(ro_batch: LongTensor2D, num_entities: int) -> LongTensor2D:
        """
        Expands relation-object pairs into all possible subject-relation-object triples.

        For each (relation, object) pair in the input batch, creates num_entities triples
        by combining the pair with every possible entity ID from 0 to num_entities-1 as subject.

        Args:
            ro_batch: Tensor of shape (batch_size, 2) containing [relation, object] pairs
            num_entities: Total number of entities to use as potential subjects

        Returns:
            Tensor of shape (batch_size * num_entities, 3) containing all possible
            [subject, relation, object] triples

        Note:
            This method assumes contiguous entity IDs from 0 to num_entities-1
        """

        batch_size = ro_batch.size(0)
        all_subjects = torch.arange(num_entities, device=ro_batch.device)
        expanded_subjects = all_subjects.unsqueeze(0).expand(batch_size, -1).flatten()

        expanded_ro = ro_batch.repeat_interleave(num_entities, dim=0)

        sro_batch = torch.empty(
            (len(expanded_subjects), 3), dtype=torch.long, device=ro_batch.device
        )
        sro_batch[:, 0] = expanded_subjects
        sro_batch[:, 1] = expanded_ro[:, 0]
        sro_batch[:, 2] = expanded_ro[:, 1]

        return sro_batch

    @staticmethod
    def generate_fact_batch_proximity_graph(
        sro_batch: LongTensor2D, num_entities: int, max_distance: int = 0
    ) -> ProximityNetworkData:
        """
        Generate a proximity graph from a batch of subject-relation-object triples.

        This method constructs a graph representing entity relationships and computes
        the distance between entities up to a specified maximum distance. It returns
        a ProximityNetworkData containing padded tensors of neighbors and their distances.

        Args:
            sro_batch (LongTensor2D): A tensor of shape (batch_size, 3) containing
                subject-relation-object triples, where each row is
                [subject_id, relation_id, object_id].
            num_entities (int): The total number of unique entities.
            max_distance (int, optional): The maximum distance to consider when finding
                higher-order neighbors. If 0, only direct (distance 1) neighbors are included.
                Default is 0.

        Returns:
            ProximityNetworkData: A dataclass containing:
                - neighbors: Padded tensor of shape (num_entities, max_neighbors) containing
                            neighbor indices for each entity (-1 for padding).
                - distances: Padded tensor of shape (num_entities, max_neighbors) containing
                            the distance to each neighbor (-1 for padding).
                - lengths: Tensor of shape (num_entities,) containing the actual number of
                        neighbors for each entity.

        Note:
            Entities are considered neighbors if they appear together in any triple,
            regardless of their role (subject or object) or the relation between them.
        """

        subjects = sro_batch[:, 0]
        objects = sro_batch[:, 2]

        # Store neighbors and distances for each entity
        neighbor_lists: list[set] = [set() for _ in range(num_entities)]
        distance_maps: list[dict] = [{} for _ in range(num_entities)]  # entity -> distance

        # First get immediate neighbors (distance 1)
        for s, o in zip(subjects, objects, strict=False):
            s_item, o_item = s.item(), o.item()
            neighbor_lists[s_item].add(o_item)
            neighbor_lists[o_item].add(s_item)
            distance_maps[s_item][o_item] = 1
            distance_maps[o_item][s_item] = 1

        # Get higher-order neighbors up to max_distance
        for dist in range(2, max_distance + 1):
            for entity in range(num_entities):
                current_neighbors = neighbor_lists[entity]
                new_neighbors = set()

                for neighbor in current_neighbors:
                    neighbors_of_neighbor = neighbor_lists[neighbor]
                    for n in neighbors_of_neighbor:
                        if n not in distance_maps[entity]:
                            new_neighbors.add(n)
                            distance_maps[entity][n] = dist

                # Remove self and already found neighbors
                new_neighbors.discard(entity)
                neighbor_lists[entity].update(new_neighbors)

        # Convert to tensors
        neighbor_tensors = []
        distance_tensors = []

        for entity in range(num_entities):
            neighbors = list(neighbor_lists[entity])
            distances = [distance_maps[entity][n] for n in neighbors]
            neighbor_tensors.append(torch.tensor(neighbors, dtype=torch.long))
            distance_tensors.append(torch.tensor(distances, dtype=torch.long))

        # Pad sequences
        padded_neighbors = torch.nn.utils.rnn.pad_sequence(
            neighbor_tensors, batch_first=True, padding_value=-1
        )
        padded_distances = torch.nn.utils.rnn.pad_sequence(
            distance_tensors, batch_first=True, padding_value=-1
        )
        lengths = torch.tensor([len(n) for n in neighbor_lists])

        return ProximityNetworkData(padded_neighbors, padded_distances, lengths)

    @staticmethod
    def to_networkx(
        sro_batch: LongTensor2D,
        num_entities: int | None = None,
        num_relations: int | None = None,
    ) -> nx.MultiDiGraph:
        graph: nx.MultiDiGraph = nx.MultiDiGraph()

        if num_entities is not None:
            graph.add_nodes_from(range(num_entities))

        triples = sro_batch.data.cpu().numpy()
        max_relation = 0
        for subj, rel, obj in triples:
            graph.add_edge(int(subj), int(obj), relation=int(rel))
            max_relation = max(max_relation, int(rel))

        for node in graph.nodes():
            graph.nodes[node]["node_index"] = node
        if num_relations is not None:
            graph.graph["num_relations"] = num_relations
        else:
            graph.graph["num_relations"] = max_relation + 1

        return graph

    @staticmethod
    def remove_batch(fact_batch: LongTensor2D, fact_batch_to_remove: LongTensor2D) -> LongTensor2D:
        mask = ~(fact_batch[:, None] == fact_batch_to_remove[None, :]).all(dim=2).any(dim=1)
        return fact_batch[mask]
