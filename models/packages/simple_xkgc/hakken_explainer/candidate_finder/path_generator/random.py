import numpy as np
import torch
from hakken_ml_toolkit.ml_base_structures.fact import FactIndex, FactIndexList
from loguru import logger

from .base import PathGenerator


class RandomPathGenerator(PathGenerator):
    def __init__(self, entity_indices: list[int], relation_indices: list[int], **kwargs) -> None:
        """Initialize the RandomPathGenerator with entity and relation indices.

        Args:
            entity_indices (list[int]): List of available entity indices.
            relation_indices (list[int]): List of available relation indices.
        """
        super().__init__(**kwargs)
        self.entity_indices = np.array(entity_indices)
        self.relation_indices = np.array(relation_indices)

    @torch.no_grad()
    def generate_facts(
        self,
        source: list[int] | torch.Tensor | None = None,
        allowed_relations: list[int] | None = None,
        target: list[int] | torch.Tensor | None = None,
        num_facts_per_entity: int = 1,
    ) -> list[FactIndex]:
        # Determine which relations to use
        relation_np: np.ndarray = (
            np.array(allowed_relations) if allowed_relations is not None else self.relation_indices
        )

        if relation_np.size == 0:
            return []

        sources: np.ndarray
        relations: np.ndarray
        targets: np.ndarray

        # Case 1: Both source and target provided - generate facts for each pair
        if source is not None and target is not None:
            source_np = self.convert_to_numpy(source)
            target_np = self.convert_to_numpy(target)
            if len(source_np) != len(target_np):
                msg = "source and target must have the same length when both are provided"
                raise ValueError(msg)

            sources = np.repeat(source_np, num_facts_per_entity)
            targets = np.repeat(target_np, num_facts_per_entity)
            relations = np.random.choice(relation_np, size=len(sources))

        elif source is not None:
            source_np = self.convert_to_numpy(source)
            sources = np.repeat(source_np, num_facts_per_entity)
            relations = np.random.choice(relation_np, size=len(sources))
            targets = np.random.choice(self.entity_indices, size=len(sources))

        # Case 3: Only target provided - generate facts for each target entity
        elif target is not None:
            target_np = self.convert_to_numpy(target)
            targets = np.repeat(target_np, num_facts_per_entity)
            relations = np.random.choice(relation_np, size=len(targets))
            sources = np.random.choice(self.entity_indices, size=len(targets))

        else:
            msg = "Either source or target must be provided"
            raise ValueError(msg)

        return list(
            zip(
                sources.astype(int).tolist(),
                relations.astype(int).tolist(),
                targets.astype(int).tolist(),
                strict=False,
            )
        )

    def generate_paths(
        self,
        source: int,
        target: int,
        num_hops: int,
        previous_generated_paths: list[FactIndexList] | None = None,
        allowed_relations: list[int] | None = None,
        num_paths: int = 1,
    ) -> list[FactIndexList]:
        relations = (
            np.array(allowed_relations) if allowed_relations is not None else self.relation_indices
        )

        if len(relations) == 0:
            return []

        previous_paths_set = set()
        if previous_generated_paths is not None:
            previous_paths_set = {tuple(path) for path in previous_generated_paths}

        # Generate more paths than needed to account for duplicates
        batch_size = max(int(num_paths * 1.2), 100)

        current_entities = np.full(batch_size, source, dtype=np.int64)

        all_facts = np.zeros((batch_size, num_hops, 3), dtype=np.int64)

        for hop in range(num_hops - 1):
            hop_relations = np.random.choice(relations, size=batch_size)
            hop_targets = np.random.choice(self.entity_indices, size=batch_size)

            all_facts[:, hop, 0] = current_entities
            all_facts[:, hop, 1] = hop_relations
            all_facts[:, hop, 2] = hop_targets

            current_entities = hop_targets

        final_relations = np.random.choice(relations, size=batch_size)
        all_facts[:, num_hops - 1, 0] = current_entities
        all_facts[:, num_hops - 1, 1] = final_relations
        all_facts[:, num_hops - 1, 2] = target

        unique_paths = []
        seen = previous_paths_set.copy()

        for path_array in all_facts.astype(int):
            path = [(int(fact[0]), int(fact[1]), int(fact[2])) for fact in path_array]
            path_tuple = tuple(path)
            if path_tuple not in seen:
                seen.add(path_tuple)
                unique_paths.append(path)
                if len(unique_paths) >= num_paths:
                    break

        if len(unique_paths) < num_paths:
            logger.warning(
                f"Only generated {len(unique_paths)}/{num_paths} unique paths "
                f"from batch of {batch_size}"
            )

        return unique_paths[:num_paths]
