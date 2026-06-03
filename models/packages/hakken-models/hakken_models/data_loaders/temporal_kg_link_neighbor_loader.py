"""Temporal-aware KG link neighbor loader for SeGAL.

Extends :class:`KGLinkNeighborLoader` to enforce a temporal constraint:
during neighbor sampling, only edges whose timestamp is strictly less
than the target timestamp ``t`` are traversed.  This guarantees that
the sampled context subgraph contains no future information relative
to the facts being scored.

Implementation strategy: the full graph is stored once, and for each
unique target timestamp the loader dynamically masks out edges that
violate the temporal constraint before sampling.

When ``group_by_timestamp=True`` (the default), a
:class:`TimestampGroupedBatchSampler` ensures that every mini-batch
contains seed edges with the **same** target timestamp.  This
eliminates the information loss caused by the ``min(batch_timestamps)``
cutoff, because within a same-timestamp batch the cutoff equals every
individual target's timestamp.
"""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import Tensor
from torch_geometric.loader import LinkNeighborLoader

from hakken_models.core.entities.kg_data import KGData
from hakken_models.data_loaders.timestamp_grouped_batch_sampler import (
    TimestampGroupedBatchSampler,
)


def corrupt_entity_pairs(
    subjects: Tensor,
    objects: Tensor,
    n_id: Tensor,
    num_negatives: int = 1,
) -> Tensor:
    """Generate negative entity pairs by corrupting subjects or objects.

    For each positive pair ``(s, o)``, generates ``num_negatives``
    corrupted pairs by randomly replacing either ``s`` or ``o`` with
    another node drawn uniformly from the subgraph nodes ``n_id``.

    Args:
        subjects: ``[B]`` global subject IDs.
        objects: ``[B]`` global object IDs.
        n_id: ``[N_sub]`` global IDs of all nodes in the subgraph.
        num_negatives: Number of negatives per positive pair.

    Returns:
        ``[2, B, K]`` tensor of ``(neg_subjects, neg_objects)`` in global IDs.
    """
    batch_size = subjects.size(0)
    num_nodes = n_id.size(0)
    device = subjects.device

    corrupt_subject = torch.randint(2, (batch_size, num_negatives), device=device).bool()

    random_nodes = n_id[torch.randint(num_nodes, (batch_size, num_negatives), device=device)]

    s_expanded = subjects.unsqueeze(1).expand_as(random_nodes)
    o_expanded = objects.unsqueeze(1).expand_as(random_nodes)
    neg_subjects = torch.where(corrupt_subject, random_nodes, s_expanded)
    neg_objects = torch.where(~corrupt_subject, random_nodes, o_expanded)

    return torch.stack([neg_subjects, neg_objects])


class TemporalKGLinkNeighborLoader(LinkNeighborLoader):
    """Link neighbor loader with edge-level temporal filtering.

    For a batch of seed edges whose target timestamp is ``t``, only
    context edges with ``timestamp <= t`` are available for neighbor
    sampling.  Target edges themselves are excluded from the
    message-passing graph to prevent data leakage.

    The loader expects ``edge_attr`` to have at least two columns:

    * column 0 — relation index
    * column 1 — timestamp index

    Args:
        data: Full knowledge graph (without ``n_id``).
        num_neighbors: Neighbors per hop for sampling.
        edge_label_index: ``[2, N]`` target entity pairs ``(subject, object)``.
        edge_label: ``[N]`` target relation indices.
        target_timestamps: ``[N]`` timestamp for each target edge.  Used
            to enforce the ``<= t`` temporal constraint on context edges.
        num_negatives: Number of corrupted entity pairs to generate per
            positive.  Stored on each batch as ``neg_edge_label_index``
            of shape ``[2, B, K]``.
        target_relation_labels: Optional ``[N, R]`` multi-hot relation
            labels precomputed by :func:`build_fact_relation_labels`.
            When provided, each batch will carry ``relation_labels``
            of shape ``[B, R]``.
        batch_size: Seed edges per batch.
        shuffle: Whether to shuffle seed edges.
        group_by_timestamp: If ``True``, use a
            :class:`TimestampGroupedBatchSampler` so that each batch
            contains only seed edges with the same target timestamp.
            This avoids the information loss from the
            ``min(batch_timestamps)`` cutoff.  When enabled, *batch_size*
            and *shuffle* configure the grouped sampler instead of being
            forwarded directly to the underlying :class:`DataLoader`.
        **kwargs: Forwarded to :class:`LinkNeighborLoader`.
    """

    def __init__(
        self,
        data: KGData,
        num_neighbors: list[int],
        edge_label_index: Tensor,
        edge_label: Tensor,
        target_timestamps: Tensor,
        num_negatives: int = 32,
        target_relation_labels: Tensor | None = None,
        batch_size: int = 1,
        shuffle: bool = True,
        group_by_timestamp: bool = True,
        **kwargs: Any,
    ) -> None:
        if data.has_n_id():
            msg = "data must not contain the original node indexes (n_id)"
            raise ValueError(msg)

        self._full_edge_attr = data.edge_attr
        self._full_edge_index = data.edge_index
        self._target_timestamps = target_timestamps
        self.num_neighbors = num_neighbors
        self._num_negatives = num_negatives
        self._target_relation_labels = target_relation_labels
        self._group_by_timestamp = group_by_timestamp

        if group_by_timestamp:
            drop_last = kwargs.pop("drop_last", False)
            kwargs.pop("sampler", None)
            batch_sampler = TimestampGroupedBatchSampler(
                timestamps=target_timestamps,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
            )
            super().__init__(
                data=data,
                num_neighbors=num_neighbors,
                edge_label_index=edge_label_index,
                edge_label=edge_label,
                neg_sampling=None,
                is_sorted=False,
                filter_per_worker=False,
                batch_sampler=batch_sampler,
                **kwargs,
            )
        else:
            super().__init__(
                data=data,
                num_neighbors=num_neighbors,
                batch_size=batch_size,
                edge_label_index=edge_label_index,
                edge_label=edge_label,
                neg_sampling=None,
                is_sorted=False,
                filter_per_worker=False,
                shuffle=shuffle,
                **kwargs,
            )

    def _postprocess(self, raw_batch: Any, batch_indices: Tensor) -> KGData:
        """Remap edge labels to global IDs, apply temporal filtering, and sample negatives.

        1. Remap ``edge_label_index`` from local to global IDs via ``n_id``.
        2. Compute the temporal cutoff (min target timestamp in this batch)
           and remove context edges whose timestamp exceeds it.
        3. Generate corrupted entity pairs and store as
           ``neg_edge_label_index`` of shape ``[2, B, K]``.
        4. Slice precomputed ``relation_labels`` for this batch (if available).
        """
        raw_batch.edge_label_index = raw_batch.n_id[raw_batch.edge_label_index]

        batch_target_ts = self._target_timestamps[batch_indices]
        edge_ts = raw_batch.edge_attr[:, 1].float()
        temporal_cutoff = batch_target_ts.min().to(edge_ts.device)
        mask = edge_ts <= temporal_cutoff
        raw_batch.edge_index = raw_batch.edge_index[:, mask]
        raw_batch.edge_attr = raw_batch.edge_attr[mask]
        if raw_batch.e_id is not None:
            raw_batch.e_id = raw_batch.e_id[mask]

        raw_batch.neg_edge_label_index = corrupt_entity_pairs(
            subjects=raw_batch.edge_label_index[0],
            objects=raw_batch.edge_label_index[1],
            n_id=raw_batch.n_id,
            num_negatives=self._num_negatives,
        )

        if self._target_relation_labels is not None:
            raw_batch.relation_labels = self._target_relation_labels[batch_indices]

        raw_batch.target_timestamps = batch_target_ts

        return cast(KGData, raw_batch)

    def __call__(self, index: Tensor | list[int]) -> KGData:
        """Sample a subgraph with temporal filtering and return it.

        The returned :class:`KGData` object has:

        * ``x``, ``edge_index``, ``edge_attr``, ``n_id`` — the sampled
          context subgraph (only edges with ``timestamp <= t``).
        * ``edge_label_index`` — ``[2, B]`` target entity pairs mapped
          to **global** node indices.
        * ``edge_label`` — ``[B]`` target relation indices.
        * ``input_id`` — ``[B]`` seed edge indices within the batch.
        * ``relation_labels`` — ``[B, R]`` multi-hot relation labels
          (if ``target_relation_labels`` was provided at construction).
        * ``target_timestamps`` — ``[B]`` target timestamp for each seed edge.
        """
        raw_batch = super().__call__(index)

        if isinstance(index, Tensor):
            batch_indices = index
        else:
            batch_indices = torch.tensor(index, dtype=torch.long)

        return self._postprocess(raw_batch, batch_indices)

    def __iter__(self):  # type: ignore[override]
        """Yield temporally-filtered batches during DataLoader iteration."""
        for raw_batch in super().__iter__():
            yield self._postprocess(raw_batch, raw_batch.input_id)
