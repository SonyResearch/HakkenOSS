from typing import cast

import numpy as np
import torch
from hakken_ml_toolkit.ml_base_structures.fact import FactIndex, FactIndexList
from hakken_ml_toolkit.ml_utils.extras import TensorCreator
from kge.models.base import KGEI
from loguru import logger

from hakken_explainer.exceptions import FactGenerationError

from .base import PathGenerator


class KGEPathGenerator(PathGenerator):
    """Generates facts and paths in knowledge graphs using a KGE model.

    This class utilizes a trained knowledge graph embedding (KGE) model to predict
    relationships between entities and generate paths of facts within a knowledge graph.
    Facts are filtered by a minimum confidence score threshold.
    """

    def __init__(
        self,
        entity_indices: list[int],
        relation_indices: list[int],
        model: KGEI,
        min_score: float = 0.7,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.entity_indices = np.array(entity_indices)
        self.relation_indices = relation_indices
        self.model = model
        self.min_score = min_score

    def to_device(self, device: str | torch.device = "cpu") -> None:
        super().to_device(device)
        self.model = self.model.to(device)

    def _get_num_facts(
        self,
        source: torch.Tensor | None,
        target: torch.Tensor | None,
    ) -> int:
        """Determine how many facts need to be generated."""
        if source is not None:
            return source.size(0)
        if target is not None:
            return target.size(0)
        raise FactGenerationError()

    def _initialize_facts_tensor(
        self,
        num_facts: int,
        source: torch.Tensor | None,
        target: torch.Tensor | None,
    ) -> torch.Tensor:
        """Create the [num_facts, 3] tensor and pre-fill known subject/object positions."""
        facts = -torch.ones((num_facts, 3), dtype=torch.long, device=self.device)

        if source is not None:
            facts[:, 0] = source
        if target is not None:
            facts[:, 2] = target

        return facts

    def _update_valid_mask(
        self,
        facts: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> int:
        """
        Score invalid facts, update the valid_mask in-place,
        and return the new number of valid facts.
        """
        invalid_mask = ~valid_mask
        if invalid_mask.sum() == 0:
            return cast("int", valid_mask.sum().item())

        invalid_facts = facts[invalid_mask]

        scores = self.model.score(invalid_facts)
        norm_scores = self.model.normalize_scores(scores).flatten()

        is_valid = norm_scores >= self.min_score
        invalid_idx_tensor = torch.where(invalid_mask)[0]
        newly_valid_indices = invalid_idx_tensor[is_valid]
        valid_mask[newly_valid_indices] = True

        return cast("int", valid_mask.sum().item())

    def _sample_entities(self, k: int, exclude: set[int] | None = None) -> list[int]:
        """Sample k entities from entity_indices without replacement.

        Args:
            k: Number of entities to sample.
            exclude: Set of entity indices to exclude from sampling. Defaults to None.

        Returns:
            List of sampled entity indices.
        """
        entities = self.entity_indices

        if exclude:
            mask = ~np.isin(entities, list(exclude), assume_unique=True)
            available = entities[mask]
        else:
            available = entities

        n = available.size
        if n == 0 or k <= 0:
            return []

        k = min(k, n)
        rng = np.random.default_rng()
        sampled = rng.choice(available, size=k, replace=False)
        return cast("list[int]", sampled.tolist())

    @torch.no_grad()
    def generate_facts(
        self,
        source: list[int] | torch.Tensor | None = None,
        allowed_relations: list[int] | None = None,
        target: list[int] | torch.Tensor | None = None,
        num_facts_per_entity: int = 1,
    ) -> list[FactIndex]:
        source_tensor: torch.Tensor | None = None
        if source is None and target is None:
            raise FactGenerationError()
        if source is not None:
            source_tensor_ = self.convert_to_torch_tensor(source)
            source_tensor = source_tensor_.repeat_interleave(num_facts_per_entity)

        target_tensor: torch.Tensor | None = None
        if target is not None:
            target_tensor_ = self.convert_to_torch_tensor(target)
            target_tensor = target_tensor_.repeat_interleave(num_facts_per_entity)

        if (
            source_tensor is not None
            and target_tensor is not None
            and source_tensor.size(0) != target_tensor.size(0)
        ):
            msg = (
                f"Source and target must have the same length after expansion. "
                f"Got source: {source_tensor.size(0)}, target: {target_tensor.size(0)}"
            )
            raise ValueError(msg)

        facts_tensor = self.generate_tensor_facts(
            source=source_tensor,
            allowed_relations=allowed_relations,
            target=target_tensor,
        )
        facts_np = facts_tensor.cpu().numpy().astype(int)
        return [tuple(fact.tolist()) for fact in facts_np]

    @torch.no_grad()
    def generate_tensor_facts(
        self,
        source: torch.Tensor | None = None,
        allowed_relations: list[int] | None = None,
        target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_facts = self._get_num_facts(source, target)

        self.model.eval()

        relations_to_use = self.relation_indices if allowed_relations is None else allowed_relations

        facts = self._initialize_facts_tensor(num_facts, source=source, target=target)

        valid_mask = torch.zeros(num_facts, dtype=torch.bool, device=self.device)

        num_valid_facts = 0
        max_iterations = 100  # Prevent infinite loops
        iteration = 0

        while num_valid_facts < num_facts and iteration < max_iterations:
            iteration += 1
            invalid_indices = ~valid_mask
            num_invalid = cast("int", invalid_indices.sum().item())
            if num_invalid == 0:
                break

            # 1) sample relations for invalid facts
            sampled_relations = np.random.choice(relations_to_use, size=num_invalid, replace=True)

            facts[invalid_indices, 1] = torch.tensor(
                sampled_relations, dtype=torch.long, device=self.device
            )

            # 2) sample missing subjects/objects (if needed)

            if source is None:
                sampled_subjects = self._sample_entities(num_invalid)
                facts[invalid_indices, 0] = torch.tensor(
                    sampled_subjects, dtype=torch.long, device=self.device
                )
            if target is None:
                sampled_objects = self._sample_entities(num_invalid)
                facts[invalid_indices, 2] = torch.tensor(
                    sampled_objects, dtype=torch.long, device=self.device
                )

            # 3) score invalid facts and update validity

            invalid_facts = facts[invalid_indices]

            scores = self.model.score(invalid_facts)
            norm_scores = self.model.normalize_scores(scores).flatten()

            is_valid = norm_scores >= self.min_score
            invalid_idx_tensor = torch.where(invalid_indices)[0]
            newly_valid_indices = invalid_idx_tensor[is_valid]
            valid_mask[newly_valid_indices] = True
            num_valid_facts = cast("int", valid_mask.sum().item())

        if num_valid_facts < num_facts:
            valid_facts = facts[valid_mask]
            num_needed = num_facts - num_valid_facts
            duplicate_indices = torch.randint(
                0, num_valid_facts, size=(num_needed,), device=self.device
            )
            duplicated_facts = valid_facts[duplicate_indices]
            facts = torch.cat([valid_facts, duplicated_facts], dim=0)

        return facts

    def generate_paths(
        self,
        source: int,
        target: int,
        num_hops: int,
        previous_generated_paths: list[FactIndexList] | None = None,
        allowed_relations: list[int] | None = None,
        num_paths: int = 1,
    ) -> list[FactIndexList]:
        if previous_generated_paths is not None:
            logger.warning("Ignoring previous_generated_paths")

        source_tensor = TensorCreator.long_tensor(
            [
                source,
            ]
            * num_paths,
            device=self.device,
        )

        all_facts = []

        for _i in range(num_hops - 1):
            facts_i = self.generate_tensor_facts(
                source=source_tensor, allowed_relations=allowed_relations
            )
            all_facts.append(facts_i)
            source_tensor = facts_i[:, 2]

        target_tensor = TensorCreator.long_tensor(
            [
                target,
            ]
            * num_paths,
            device=self.device,
        )
        facts_k = self.generate_tensor_facts(
            source=source_tensor, allowed_relations=allowed_relations, target=target_tensor
        )
        all_facts.append(facts_k)

        stacked_facts = torch.stack(all_facts, dim=0)  # Shape: [num_hops, num_paths, 3]
        pathways_tensor = stacked_facts.permute(1, 0, 2)  # Shape: [num_paths, num_hops, 3]

        pathways_np: np.ndarray = pathways_tensor.cpu().numpy().astype(int)
        return [[tuple(fact.tolist()) for fact in path] for path in pathways_np]
