from __future__ import annotations

import os
from typing import TYPE_CHECKING, cast

import torch
from hakken_ml_toolkit.ml_utils.extras.scalers import StandardScaler
from loguru import logger
from torch import Tensor
from torch_geometric.loader import LinkNeighborLoader

from kge.common.entities import KGData, KGPredictionSubgraph
from kge.data_loaders.mimic_kge.config import MimicKGEDataLoaderConfig
from kge.models.kge_api import KGEAPI
from kge.negative_sampler.simple import negative_sampler

if TYPE_CHECKING:
    from collections.abc import Iterator


BatchType = tuple[KGPredictionSubgraph, Tensor, Tensor | None, Tensor, Tensor | None]


class MimicKGEDataLoader(LinkNeighborLoader):
    def __init__(
        self,
        data: KGData,
        trained_kge: KGEAPI,
        num_relations: int,
        num_neighbors: list[int],
        edge_label_index: Tensor | None = None,
        edge_label: Tensor | None = None,
        batch_size: int = 1,
        num_batches_for_scaling: int = 10,
        negs_per_pos: int = 1,
        corrupt_probs: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
        shuffle: bool = True,
        scaler_path: str | None = None,
        **kwargs,
    ) -> None:
        """Initialize MimicKGEDataLoader.

        Args:
            data: Graph data with edge_type attribute
            trained_kge: Pre-trained KGE model for scoring
            num_relations: Total number of relation types
            num_neighbors: Number of neighbors to sample per layer
            batch_size: Batch size for sampling
            device: Device to use for computations
            num_batches_for_scaling: Number of batches for fitting score scaler
            negs_per_pos: Number of negative samples per positive sample
            corrupt_probs: Probabilities for corrupting (subject, relation, object)
            shuffle: Whether to shuffle samples

        Raises:
            ValueError: If data doesn't have required edge_type attribute
        """

        self.trained_kge = trained_kge
        self.num_relations = num_relations

        self.num_batches_for_scaling = num_batches_for_scaling
        self.negs_per_pos = negs_per_pos
        self.corrupt_probs = corrupt_probs
        self.score_scaler = StandardScaler()
        self.scaler_path = scaler_path
        self._is_scaler_fitted = False

        if edge_label_index is None:
            edge_label_index = data.edge_index

        if edge_label is None:
            edge_label = data.edge_type
        super().__init__(
            data=data,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            edge_label_index=edge_label_index,
            edge_label=edge_label,
            neg_sampling_ratio=0.0,
            shuffle=shuffle,
            **kwargs,
        )

    def to_facts(
        self,
        sub_ids_local: Tensor,
        rel_ids: Tensor,
        obj_ids_local: Tensor,
        node_ids: Tensor | None = None,
    ) -> Tensor:
        """Convert local indices to global fact tensors."""

        batch_size = sub_ids_local.size(0)
        facts = torch.empty((batch_size, 3), dtype=torch.long)
        sub_ids = node_ids[sub_ids_local] if node_ids is not None else sub_ids_local
        obj_ids = node_ids[obj_ids_local] if node_ids is not None else obj_ids_local

        facts[:, 0] = sub_ids
        facts[:, 1] = rel_ids
        facts[:, 2] = obj_ids

        return facts

    def load_scaler(self) -> bool:
        success = False
        if self.scaler_path and os.path.exists(self.scaler_path):
            logger.info(f"Loading scaler from {self.scaler_path}")
            self.score_scaler = StandardScaler.load(self.scaler_path)
            self._is_scaler_fitted = True
            success = True

        if not success:
            logger.info(f"No existing scaler found at {self.scaler_path}")
        return success

    def fit_scaler(self) -> None:
        """Fit the score scaler using batched data.

        Args:
            scaler_json_path: Path to save/load scaler parameters
        """

        if self.load_scaler():
            return

        logger.info(
            f"Fitting scaler using {self.num_batches_for_scaling} batches with "
            f"{self.negs_per_pos} negatives per positive sample"
        )

        def batch_iterator() -> Iterator[torch.Tensor]:
            for batch_list in super(MimicKGEDataLoader, self).__iter__():
                batch = cast("KGPredictionSubgraph", batch_list[0])

                num_subgraph_nodes = batch.node_ids.size(0)
                src_local, dst_local = batch.edge_label_index  # local indices

                pos_rel = batch.edge_label.long()

                neg_src, neg_rel, neg_dst = negative_sampler(
                    pos_s=src_local,
                    pos_r=pos_rel,
                    pos_o=dst_local,
                    num_nodes=num_subgraph_nodes,
                    num_relations=self.num_relations,
                    device="cpu",
                    negs_per_pos=self.negs_per_pos,
                    corrupt_probs=self.corrupt_probs,
                )

                facts_pos = self.to_facts(
                    sub_ids_local=src_local,
                    rel_ids=pos_rel,
                    obj_ids_local=dst_local,
                    node_ids=batch.node_ids,
                )
                facts_neg = self.to_facts(
                    sub_ids_local=neg_src,
                    rel_ids=neg_rel,
                    obj_ids_local=neg_dst,
                    node_ids=batch.node_ids,
                )

                facts = torch.concat([facts_pos, facts_neg], dim=0)
                yield self.trained_kge.score(facts)

        self.score_scaler.fit_from_iterator(
            batch_iterator(), num_batches=self.num_batches_for_scaling
        )
        self._is_scaler_fitted = True
        if self.scaler_path:
            self.score_scaler.save(json_path=self.scaler_path)
            logger.info(f"Fitted scaler saved to {self.scaler_path}")

    def collate_fn(self, index: Tensor | list[int]) -> BatchType:
        """Collate function that processes batched subgraph data for MimicKGE prediction.

        This method transforms raw PyTorch Geometric batch data into a structured format
        suitable for MimicKGE. It extracts positive facts from the batch, generates negative
        samples, and computes KGE scores for both positive and negative samples using
        the pre-trained KGE model.

        The method performs the following key operations:
        1. Converts raw batch to KGPredictionSubgraph format
        2. Extracts positive facts (subject, relation, object triples) from edge labels
        3. Generates negative samples via corruption if negs_per_pos > 0
        4. Computes KGE scores for positive (and negative) facts using the trained model
        5. Applies score scaling if the scaler has been fitted

        Args:
            index: Batch indices as either a Tensor or list of integers specifying
                which samples to include in the current batch.

        Returns:
            A tuple containing:
                - batch (KGPredictionSubgraph): Processed subgraph with node mappings and structure
                - facts_pos (Tensor): Positive facts as [N, 3] tensor (subject, relation, object)
                using local node indices within the subgraph
                - facts_neg (Tensor | None): Negative facts as [N*negs_per_pos, 3] tensor using
                local indices, or None if negs_per_pos == 0
                - scores_pos (Tensor): KGE scores for positive facts as [N, 1] tensor,
                optionally scaled if scaler is fitted
                - scores_neg (Tensor | None): KGE scores for negative facts as [N*negs_per_pos, 1]
                tensor, or None if no negatives generated

        Note:
            - Local indices in facts_pos and facts_neg refer to positions within the
            subgraph's node_ids tensor, not global node IDs
            - Global node IDs are used internally for KGE scoring via batch.node_ids mapping
            - Score scaling is applied only if the scaler has been fitted via fit_scaler()
            - Negative sampling uses the configured corruption probabilities and negs_per_pos ratio
        """
        raw_batch = super().collate_fn(index)

        batch = KGPredictionSubgraph.from_data(raw_batch)

        num_subgraph_nodes = batch.node_ids.size(0)
        src_local, dst_local = batch.edge_label_index  # local indices
        pos_rel = batch.edge_label.long()

        facts_pos = self.to_facts(
            sub_ids_local=src_local,
            rel_ids=pos_rel,
            obj_ids_local=dst_local,
        )

        facts_global_pos = self.to_facts(
            sub_ids_local=src_local,
            rel_ids=pos_rel,
            obj_ids_local=dst_local,
            node_ids=batch.node_ids,
        )

        scores_pos = self.trained_kge.score(facts_global_pos).unsqueeze(-1)

        if self._is_scaler_fitted:
            scores_pos = self.score_scaler.transform(scores_pos)

        facts_neg: Tensor | None = None
        scores_neg: Tensor | None = None

        if self.negs_per_pos > 0:
            neg_src, neg_rel, neg_dst = negative_sampler(
                pos_s=src_local,
                pos_r=pos_rel,
                pos_o=dst_local,
                num_nodes=num_subgraph_nodes,
                num_relations=self.num_relations,
                device="cpu",
                negs_per_pos=self.negs_per_pos,
                corrupt_probs=self.corrupt_probs,
            )

            facts_neg = self.to_facts(
                sub_ids_local=neg_src,
                rel_ids=neg_rel,
                obj_ids_local=neg_dst,
            )

            facts_global_neg = self.to_facts(
                sub_ids_local=neg_src,
                rel_ids=neg_rel,
                obj_ids_local=neg_dst,
                node_ids=batch.node_ids,
            )
            scores_neg = self.trained_kge.score(facts_global_neg).unsqueeze(-1)
            if self._is_scaler_fitted:
                scores_neg = self.score_scaler.transform(scores_neg)

        return batch, facts_pos, facts_neg, scores_pos, scores_neg

    @classmethod
    def from_config(
        cls,
        config: MimicKGEDataLoaderConfig,
        data: KGData,
        trained_kge: KGEAPI,
        subgraph_data: KGData | None = None,
    ) -> MimicKGEDataLoader:
        """Create a MimicKGEDataLoader from configuration.

        This method creates a data loader that separates the graph structure used for
        message passing from the facts/edges used for link prediction tasks.


        Args:
            config: Configuration object containing all settings for the data loader
            data: **Prediction target data** - Contains the facts (edge_index, edge_type)
                where we want to make link predictions. These edges define the positive
                samples and are used to generate negative samples for training/evaluation.
                This data provides the ground truth labels for the prediction task.
            trained_kge: Pre-trained KGE model for scoring facts
            subgraph_data: **Message passing data** - Optional separate graph structure
                used for neighborhood sampling and GNN message passing. If None, defaults
                to `data`. This allows using a different (potentially larger or filtered)
                graph topology for computing node representations while predicting on a
                specific set of target edges from `data`.

        Returns:
            Configured MimicKGEDataLoader instance
        """

        if subgraph_data is None:
            subgraph_data = data

        return cls(
            data=subgraph_data,
            trained_kge=trained_kge,
            num_relations=config.num_relations,
            num_neighbors=config.num_neighbors,
            edge_label_index=data.edge_index,
            edge_label=data.edge_type,
            batch_size=config.batch_size,
            num_batches_for_scaling=config.num_batches_for_scaling,
            negs_per_pos=config.negs_per_pos,
            corrupt_probs=config.corrupt_probs,
            shuffle=config.shuffle,
            scaler_path=config.scaler_path,
            pin_memory=config.pin_memory,
            num_workers=config.num_workers,
        )
