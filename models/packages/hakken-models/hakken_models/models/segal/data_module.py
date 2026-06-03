"""SeGAL Lightning DataModule backed by TemporalKGLinkNeighborLoader.

Constructs :class:`KGData` from indexed fact tensors, splits target edges
by timestamp, and produces temporal-aware batches where the context
subgraph only contains edges with ``timestamp <= min(t_targets)``.
"""

from __future__ import annotations

import torch
from lightning.pytorch import LightningDataModule
from loguru import logger
from torch import Tensor
from torch.utils.data import DataLoader

from hakken_models.core.entities.kg_data import KGData
from hakken_models.data_loaders import TemporalKGLinkNeighborLoader

_COL_S = 0
_COL_R = 1
_COL_O = 2
_COL_T = 3


class SeGALDataModule(LightningDataModule):
    """Lightning DataModule for SeGAL training and validation.

    Builds :class:`KGData` graphs and uses
    :class:`TemporalKGLinkNeighborLoader` to sample temporal context
    subgraphs around target entity pairs.

    The **training** context graph is built from ``train_facts`` only.
    The **validation** context graph is built from ``train_facts + val_facts``
    because at inference time the model has access to all known facts;
    temporal filtering (``<= min(t_targets)``) already prevents leakage.

    The loader yields :class:`KGData` batches where:

    * ``edge_index`` / ``edge_attr`` contain only edges with timestamps
      ``<= min(target_timestamps)`` in the batch.
    * ``edge_label_index`` contains target ``(subject, object)`` pairs
      mapped to **global** node IDs.
    * ``edge_label`` contains the target **relation indices**.
    * ``relation_labels`` (when provided) contains ``[B, R]`` multi-hot
      vectors encoding which relations are known to hold between each
      target entity pair at the target timestamp.

    Args:
        train_facts: ``[num_train, 4]`` integer fact tensor
            ``(s, r, o, t_idx)`` for training.
        val_facts: ``[num_val, 4]`` integer fact tensor for validation.
        num_nodes: Total number of nodes in the knowledge graph.
        train_relation_labels: Optional precomputed ``[num_train, R]``
            multi-hot relation labels for the training split.
        val_relation_labels: Optional precomputed ``[num_val, R]``
            multi-hot relation labels for the validation split.
        num_neighbors: Neighbors per hop for the GNN sampler.
        batch_size: Seed edges per batch.
        num_negatives: Number of corrupted entity pairs per positive
            during **training**.
        num_negatives_val: Number of corrupted entity pairs per positive
            during **validation**.  Falls back to ``num_negatives`` when
            ``None``.
        add_reverse_edges: If True, build context graphs with reverse edges
            (object→subject) and a direction column in ``edge_attr`` (0/1).
            Defaults to True for SeGAL inverse-relation support.
        **kwargs: Forwarded to the underlying data loader (e.g.
            ``num_workers``, ``pin_memory``).
    """

    def __init__(
        self,
        train_facts: Tensor,
        val_facts: Tensor,
        num_nodes: int,
        train_relation_labels: Tensor | None = None,
        val_relation_labels: Tensor | None = None,
        num_neighbors: list[int] | None = None,
        batch_size: int = 32,
        num_negatives: int = 32,
        num_negatives_val: int | None = None,
        add_reverse_edges: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        if num_neighbors is None:
            num_neighbors = [128, 128]

        self.num_neighbors = num_neighbors
        self.batch_size = batch_size
        self.num_negatives = num_negatives
        self.num_negatives_val = (
            num_negatives_val if num_negatives_val is not None else num_negatives
        )
        self.kwargs = kwargs

        self.train_kg_data = KGData.from_facts(
            train_facts,
            num_nodes=num_nodes,
            relabel_nodes=False,
            sort=True,
            add_reverse_edges=add_reverse_edges,
        )

        all_facts = torch.cat([train_facts, val_facts], dim=0)
        self.val_kg_data = KGData.from_facts(
            all_facts,
            num_nodes=num_nodes,
            relabel_nodes=False,
            sort=True,
            add_reverse_edges=add_reverse_edges,
        )

        self.train_entity_pairs = train_facts[:, [_COL_S, _COL_O]]
        self.train_relations = train_facts[:, _COL_R]
        self.train_timestamps = train_facts[:, _COL_T].float()

        self.val_entity_pairs = val_facts[:, [_COL_S, _COL_O]]
        self.val_relations = val_facts[:, _COL_R]
        self.val_timestamps = val_facts[:, _COL_T].float()

        self.train_relation_labels = train_relation_labels
        self.val_relation_labels = val_relation_labels

    def train_dataloader(self) -> DataLoader:
        kwargs = self.kwargs.copy()
        kwargs.update({"shuffle": True, "batch_size": self.batch_size})

        logger.info(f"Train timestamps: {self.train_timestamps.unique()}")

        return TemporalKGLinkNeighborLoader(
            data=self.train_kg_data,
            num_neighbors=self.num_neighbors,
            edge_label_index=self.train_entity_pairs.t().contiguous(),
            edge_label=self.train_relations,
            target_timestamps=self.train_timestamps,
            num_negatives=self.num_negatives,
            target_relation_labels=self.train_relation_labels,
            **kwargs,
        )

    def val_dataloader(self) -> DataLoader:
        kwargs = self.kwargs.copy()
        kwargs.update({"shuffle": True, "batch_size": self.batch_size})

        logger.info(f"Val timestamps: {self.val_timestamps.unique()}")

        return TemporalKGLinkNeighborLoader(
            data=self.val_kg_data,
            num_neighbors=self.num_neighbors,
            edge_label_index=self.val_entity_pairs.t().contiguous(),
            edge_label=self.val_relations,
            target_timestamps=self.val_timestamps,
            num_negatives=self.num_negatives_val,
            target_relation_labels=self.val_relation_labels,
            **kwargs,
        )
