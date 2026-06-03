from abc import ABC, abstractmethod
from enum import IntEnum
from functools import cached_property

import torch
from kge.models.gnn import GNNKGE
from loguru import logger
from torch import Tensor
from torch_geometric.utils import k_hop_subgraph

MEMORY_EFFICIENT_THRESHOLD = 1_000_000
NUM_DIMS_CANDIDATES = 3


class KGTensorSpec(IntEnum):
    """Knowledge graph tensor specifications."""

    # Tensor shapes
    FACT_BATCH_DIMS = 2  # Dimensions for fact batches [num_facts, 3]
    FACT_SIZE = 3  # Elements per fact [subject, relation, object]
    CANDIDATE_PATHS_DIMS = 3  # Dimensions for paths [num_paths, path_length, 3]

    # Triple element positions
    SUBJECT_POS = 0
    RELATION_POS = 1
    OBJECT_POS = 2


class ExplainerScore(ABC):
    """Abstract base class for scoring explanation paths in knowledge graph completion.

    This class provides a framework for evaluating how well different explanation paths
    support predictions in knowledge graph completion tasks.

    Attributes:
        context_kg (torch.Tensor): The full knowledge graph used as context for explanations.
            Shape: [num_facts, 3] where each row represents a fact [subject, relation, object].
        model (GNNKGE): The trained knowledge graph embedding model used for scoring predictions.
    """

    def __init__(self, context_kg: Tensor, model: GNNKGE) -> None:
        if (
            context_kg.dim() != KGTensorSpec.FACT_BATCH_DIMS
            or context_kg.shape[1] != KGTensorSpec.FACT_SIZE
        ):
            msg = f"context_kg must have shape [num_facts, 3], got {context_kg.shape}"
            raise ValueError(msg)
        if context_kg.numel() == 0:
            msg = "context_kg cannot be empty"
            raise ValueError(msg)

        self.context_kg = context_kg
        self.model = model

    @cached_property
    def edge_index(self) -> Tensor:
        return self.context_kg[:, [0, 2]].t().contiguous()

    def _prepare_model_and_device(self, device: torch.device) -> GNNKGE:
        """Common model preparation logic."""
        model = self.model.to(device)
        model.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.context_kg = self.context_kg.to(device)
        return model

    def get_context_kg(
        self, target_fact: Tensor, k: int, max_num_neighbors: int | None = None
    ) -> Tensor:
        """Get the relevant context knowledge graph for a target fact.

        Extracts a k-hop subgraph around the subject and object entities of the target fact
        to provide contextual information for scoring explanation paths. Uses PyTorch Geometric's
        k_hop_subgraph function.

            Args:
                target_fact (torch.Tensor): Target fact to explain. Shape: [1, 3] representing
                    a single fact with [subject, relation, object] entity IDs.
                k (int): Number of hops to expand around target entities. Higher values include
                    more distant neighbors but increase computational cost.
                max_num_neighbors (int, optional): Maximum number of neighbor entities to include
                    in the context subgraph. Currently not implemented - raises NotImplementedError
                    if provided.

            Returns:
                torch.Tensor: Filtered knowledge graph containing only facts relevant to the
                    target entities within k hops, excluding the target fact itself.
                    Shape: [num_relevant_facts, 3] where each row is a fact
                    [subject, relation, object]. Preserves original entity IDs from the
                    input knowledge graph.
            Raises:
                NotImplementedError: If max_num_neighbors is provided (feature not yet implemented).

            Note:
                - Automatically excludes the target fact from the returned context
                - All facts in the returned subgraph have both subject and object within
                the k-hop neighborhood of the target entities
        """

        if target_fact.dim() != KGTensorSpec.FACT_BATCH_DIMS or target_fact.shape != torch.Size(
            [1, 3]
        ):
            msg = f"target_fact must have shape [1, 3], got {target_fact.shape}"
            raise ValueError(msg)
        if k <= 0:
            msg = f"k must be positive, got {k}"
            raise ValueError()

        if max_num_neighbors is not None:
            raise NotImplementedError()

        node_idx = target_fact[0, [0, 2]].tolist()

        # Create edge index from knowledge graph (bidirectional)

        _subset, _edge_index, _mapping, edge_mask = k_hop_subgraph(
            node_idx=node_idx, num_hops=k, edge_index=self.edge_index
        )

        context_kg = self.context_kg[edge_mask]

        mask_not_target = ~torch.all(context_kg == target_fact.to(context_kg.device), dim=1)
        return context_kg[mask_not_target]

    def filter_context_facts(
        self,
        context_facts: Tensor,
        candidate_paths: Tensor,
        max_context_facts: int,
    ) -> Tensor:
        """Remove random facts from context while preserving facts in candidate paths.

        This function ensures that all facts present in candidate_paths are retained
        in the filtered context, while randomly sampling other facts to meet the
        max_context_facts limit. Duplicate facts in candidate_paths are automatically
        deduplicated.

        Args:
            context_facts: Tensor of shape (N, 3) containing all context facts.
                        Each row is [head, relation, tail].
            candidate_paths: Tensor of shape (M, path_length, 3) or (M, 3) containing
                            candidate path facts that must be preserved.
            max_context_facts: Maximum number of facts to retain in context.

        Returns:
            Filtered context facts tensor of shape (K, 3) where K <= max_context_facts.
            All unique facts from candidate_paths are guaranteed to be included.

        Raises:
            ValueError: If max_context_facts is less than the number of unique facts
                    in candidate_paths.
        """
        if context_facts.shape[0] <= max_context_facts:
            logger.debug(
                f"Context size {context_facts.shape[0]} <= max {max_context_facts}, "
                "no filtering needed"
            )
            return context_facts

        device = context_facts.device

        if candidate_paths.dim() == NUM_DIMS_CANDIDATES:
            all_candidate_facts = candidate_paths.reshape(-1, 3)
        else:
            all_candidate_facts = candidate_paths

        if all_candidate_facts.numel() == 0:
            logger.warning("No valid candidate facts found, performing random sampling")
            indices = torch.randperm(context_facts.shape[0], device=device)[:max_context_facts]
            return context_facts[indices]

        candidate_facts: Tensor = torch.unique(all_candidate_facts, dim=0)
        logger.info(f"Unique candidate facts after deduplication: {candidate_facts.shape[0]}")

        batch_size = 1000  # Adjust based on your memory constraints

        num_context = context_facts.size(0)
        is_candidate_fact = torch.zeros(num_context, dtype=torch.bool, device=context_facts.device)

        for i in range(0, num_context, batch_size):
            end_idx = min(i + batch_size, num_context)
            context_batch = context_facts[i:end_idx]

            # Compare this batch against all candidate_facts
            matches = (context_batch.unsqueeze(1) == candidate_facts.unsqueeze(0)).all(dim=2)
            is_candidate_fact[i:end_idx] = matches.any(dim=1)

        must_keep_indices = torch.where(is_candidate_fact)[0]
        can_remove_indices = torch.where(~is_candidate_fact)[0]

        num_must_keep = must_keep_indices.shape[0]
        num_can_remove = can_remove_indices.shape[0]

        logger.info(
            f"Must keep {num_must_keep} facts from candidates, "
            f"{num_can_remove} facts can be removed"
        )

        # Check if we can meet the constraint
        if num_must_keep > max_context_facts:
            msg = (
                f"Cannot filter to {max_context_facts} facts: "
                f"{num_must_keep} unique facts from candidate_paths must be preserved. "
                f"Increase max_context_facts to at least {num_must_keep}."
            )
            raise ValueError(msg)

        num_additional = max_context_facts - num_must_keep

        if num_additional >= num_can_remove:
            logger.debug(
                f"Can keep all {num_can_remove} additional facts (limit allows {num_additional})"
            )
            return context_facts

        sampled_indices = can_remove_indices[
            torch.randperm(num_can_remove, device=device)[:num_additional]
        ]

        final_indices = torch.cat([must_keep_indices, sampled_indices])

        filtered_facts = context_facts[final_indices]

        logger.info(
            f"Filtered context from {context_facts.shape[0]} to {filtered_facts.shape[0]} facts "
            f"({num_must_keep} from candidates + {num_additional} random)"
        )

        return filtered_facts

    @abstractmethod
    @torch.no_grad()
    def score(
        self,
        target_fact: Tensor,
        candidate_paths: Tensor,
        batch_size: int = 1,
        num_hops: int = 2,
        normalize_by_original: bool = False,
    ) -> list[float]:
        """Score explanation paths for their ability to support a target fact prediction.

        This abstract method must be implemented by subclasses to define specific scoring
        strategies for evaluating how well explanation paths justify knowledge graph predictions.

        Args:
            target_fact (torch.Tensor): Target fact to explain. Shape: [1, 3]
                representing [subject, relation, object] entity IDs.
            candidate_paths (torch.Tensor): Tensor of explanation paths to score.
                Shape: [num_paths, path_length, 3] where each path is a sequence of facts
                that potentially explain the target fact.
            batch_size (int, optional): Number of paths to process in each batch for memory
                efficiency. Defaults to 1.
            normalize_by_original (bool, optional): Whether to normalize scores relative to
                a baseline (e.g., full-context prediction). Implementation depends on subclass.
                Defaults to False.

        Returns:
            list[float]: Scores for each explanation path. Higher scores indicate
                better explanations.

        Note:
            - Method is decorated with @torch.no_grad() to disable gradient computation
            - Subclasses should handle GPU memory management and batching appropriately
            - Empty candidate_paths should be handled gracefully by returning empty list
        """
        pass

    def remove_explanation_from_context(
        self, context_facts: torch.Tensor, explanation: torch.Tensor
    ) -> torch.Tensor:
        """Remove explanation facts from context facts.

        Args:
            context_facts: Tensor of shape [m, 3] containing context facts
            explanation: Tensor of shape [n, 3] containing explanation facts to remove

        Returns:
            Tensor of remaining context facts after removing explanation facts
        """

        if explanation.numel() == 0 or context_facts.numel() == 0:
            return context_facts

        if context_facts.shape[0] * explanation.shape[0] > MEMORY_EFFICIENT_THRESHOLD:
            return self._remove_explanation_chunked(context_facts, explanation)

        explanation = explanation.to(context_facts.device)

        matches = torch.all(context_facts.unsqueeze(1) == explanation.unsqueeze(0), dim=2)

        facts_to_remove = torch.any(matches, dim=1)

        return context_facts[~facts_to_remove]

    def _remove_explanation_chunked(self, context_facts: Tensor, explanation: Tensor) -> Tensor:
        """Chunked processing for memory efficiency."""
        explanation = explanation.to(context_facts.device)
        keep_mask = torch.ones(
            context_facts.shape[0], dtype=torch.bool, device=context_facts.device
        )

        for expl_fact in explanation:
            matches = torch.all(context_facts == expl_fact.unsqueeze(0), dim=1)
            keep_mask &= ~matches

        return context_facts[keep_mask]
