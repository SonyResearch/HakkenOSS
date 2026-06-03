"""Batch sampler that groups seed-edge indices by target timestamp.

When used with :class:`TemporalKGLinkNeighborLoader`, every mini-batch
contains edges that share the same target timestamp.  This eliminates
the information loss caused by the ``min(batch_timestamps)`` temporal
cutoff, because within a same-timestamp batch the cutoff equals every
individual target's timestamp.

In distributed training the sampler deterministically shards batches
across ranks while preserving the timestamp-grouping invariant within
every individual batch.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import torch
import torch.distributed as dist
from loguru import logger
from torch import Tensor
from torch.utils.data import Sampler


class TimestampGroupedBatchSampler(Sampler[list[int]]):
    """Yield batches of indices grouped by target timestamp.

    Indices are first partitioned into groups that share the same
    timestamp value.  Each epoch, the order of groups (and the order of
    indices within each group) is shuffled when ``shuffle=True``.
    Batches are then carved from each group sequentially.

    In distributed training, batches are deterministically sharded
    across ranks (round-robin) so each process sees a disjoint subset.
    The batch list is padded so every rank yields the same count.

    Args:
        timestamps: ``[N]`` target timestamp for each seed edge.
        batch_size: Maximum number of seed edges per batch.
        shuffle: If ``True``, shuffle group order and intra-group order
            every epoch.
        drop_last: If ``True``, drop the last incomplete batch in each
            group.
        generator: Optional :class:`torch.Generator` for reproducible
            shuffling (single-process only; ignored in distributed mode
            where an epoch-based seed is used instead so that all ranks
            agree on the ordering).
        num_replicas: Number of distributed processes.  Auto-detected
            from :mod:`torch.distributed` when ``None``.
        rank: Rank of the current process.  Auto-detected when ``None``.
    """

    def __init__(
        self,
        timestamps: Tensor,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        generator: torch.Generator | None = None,
        num_replicas: int | None = None,
        rank: int | None = None,
    ) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.generator = generator
        self.epoch = 0

        if num_replicas is None or rank is None:
            if dist.is_available() and dist.is_initialized():
                num_replicas = num_replicas or dist.get_world_size()
                rank = rank if rank is not None else dist.get_rank()
            else:
                num_replicas = 1
                rank = 0

        self.num_replicas = num_replicas
        self.rank = rank

        unique_ts, inverse = timestamps.unique(sorted=True, return_inverse=True)
        self._groups: list[Tensor] = [
            (inverse == i).nonzero(as_tuple=False).squeeze(1) for i in range(len(unique_ts))
        ]
        logger.info(f"Number of groups: {len(self._groups)}")

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for deterministic shuffling in distributed mode.

        All ranks must call this with the same value before iterating.
        When not called explicitly, the sampler auto-increments the
        epoch on each :meth:`__iter__` call in distributed mode.
        """
        self.epoch = epoch

    def _generate_all_batches(self, generator: torch.Generator | None) -> list[list[int]]:
        """Build the full (non-sharded) list of batches."""
        all_batches: list[list[int]] = []
        group_order = list(range(len(self._groups)))
        if self.shuffle:
            perm = torch.randperm(len(group_order), generator=generator)
            group_order = [group_order[i] for i in perm]

        for g_idx in group_order:
            indices = self._groups[g_idx]
            if self.shuffle:
                perm = torch.randperm(len(indices), generator=generator)
                indices = indices[perm]

            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size].tolist()
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                all_batches.append(batch)

        return all_batches

    def __iter__(self) -> Iterator[list[int]]:
        if self.num_replicas == 1:
            yield from self._generate_all_batches(self.generator)
            return

        g = torch.Generator()
        g.manual_seed(self.epoch)
        self.epoch += 1

        all_batches = self._generate_all_batches(g)

        per_rank = math.ceil(len(all_batches) / self.num_replicas)
        padded_total = per_rank * self.num_replicas
        if padded_total > len(all_batches):
            all_batches += all_batches[: padded_total - len(all_batches)]

        for i in range(self.rank, padded_total, self.num_replicas):
            yield all_batches[i]

    @property
    def _total_batches(self) -> int:
        if self.drop_last:
            return sum(len(g) // self.batch_size for g in self._groups)
        return sum((len(g) + self.batch_size - 1) // self.batch_size for g in self._groups)

    def __len__(self) -> int:
        return math.ceil(self._total_batches / self.num_replicas)
