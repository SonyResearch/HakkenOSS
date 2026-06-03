"""Tests for TimestampGroupedBatchSampler.

Validates grouping correctness, epoch coverage, shuffle behaviour,
batch sizing, drop_last, and edge cases.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from hakken_models.data_loaders.timestamp_grouped_batch_sampler import (
    TimestampGroupedBatchSampler,
)

# ============================================================================
# Helpers
# ============================================================================


def _collect_all_batches(sampler: TimestampGroupedBatchSampler) -> list[list[int]]:
    return list(sampler)


def _flatten_batches(batches: list[list[int]]) -> list[int]:
    return [idx for batch in batches for idx in batch]


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture()
def timestamps_10_uniform() -> Tensor:
    """100 seed edges with 10 unique timestamps (≈10 edges each)."""
    torch.manual_seed(42)
    return torch.randint(0, 10, (100,)).float()


@pytest.fixture()
def timestamps_single() -> Tensor:
    """All edges share the same timestamp."""
    return torch.full((50,), 5.0)


@pytest.fixture()
def timestamps_unique_per_edge() -> Tensor:
    """Every edge has a unique timestamp (worst case for grouping)."""
    return torch.arange(20).float()


# ============================================================================
# Coverage: every index appears exactly once per epoch
# ============================================================================


class TestEpochCoverage:
    __test__ = True

    def test_all_indices_yielded_once(self, timestamps_10_uniform: Tensor) -> None:
        sampler = TimestampGroupedBatchSampler(timestamps_10_uniform, batch_size=8, shuffle=False)
        flat = _flatten_batches(_collect_all_batches(sampler))
        assert sorted(flat) == list(range(len(timestamps_10_uniform)))

    def test_all_indices_yielded_once_with_shuffle(self, timestamps_10_uniform: Tensor) -> None:
        sampler = TimestampGroupedBatchSampler(timestamps_10_uniform, batch_size=8, shuffle=True)
        flat = _flatten_batches(_collect_all_batches(sampler))
        assert sorted(flat) == list(range(len(timestamps_10_uniform)))

    def test_all_indices_yielded_single_timestamp(self, timestamps_single: Tensor) -> None:
        sampler = TimestampGroupedBatchSampler(timestamps_single, batch_size=16, shuffle=False)
        flat = _flatten_batches(_collect_all_batches(sampler))
        assert sorted(flat) == list(range(len(timestamps_single)))

    def test_all_indices_yielded_unique_per_edge(self, timestamps_unique_per_edge: Tensor) -> None:
        sampler = TimestampGroupedBatchSampler(
            timestamps_unique_per_edge, batch_size=4, shuffle=False
        )
        flat = _flatten_batches(_collect_all_batches(sampler))
        assert sorted(flat) == list(range(len(timestamps_unique_per_edge)))


# ============================================================================
# Grouping: each batch must contain only same-timestamp indices
# ============================================================================


class TestTimestampHomogeneity:
    __test__ = True

    def test_batches_have_same_timestamp(self, timestamps_10_uniform: Tensor) -> None:
        sampler = TimestampGroupedBatchSampler(timestamps_10_uniform, batch_size=4, shuffle=False)
        for batch in sampler:
            ts_in_batch = timestamps_10_uniform[batch]
            assert (ts_in_batch == ts_in_batch[0]).all(), (
                f"Batch contains mixed timestamps: {ts_in_batch.unique().tolist()}"
            )

    def test_batches_have_same_timestamp_with_shuffle(self, timestamps_10_uniform: Tensor) -> None:
        sampler = TimestampGroupedBatchSampler(timestamps_10_uniform, batch_size=4, shuffle=True)
        for batch in sampler:
            ts_in_batch = timestamps_10_uniform[batch]
            assert (ts_in_batch == ts_in_batch[0]).all()


# ============================================================================
# Batch sizing
# ============================================================================


class TestBatchSizing:
    __test__ = True

    def test_batch_size_respected(self, timestamps_10_uniform: Tensor) -> None:
        bs = 8
        sampler = TimestampGroupedBatchSampler(timestamps_10_uniform, batch_size=bs, shuffle=False)
        for batch in sampler:
            assert len(batch) <= bs

    def test_batch_size_one(self, timestamps_10_uniform: Tensor) -> None:
        sampler = TimestampGroupedBatchSampler(timestamps_10_uniform, batch_size=1, shuffle=False)
        for batch in sampler:
            assert len(batch) == 1

    def test_batch_size_larger_than_group(self) -> None:
        ts = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2]).float()
        sampler = TimestampGroupedBatchSampler(ts, batch_size=100, shuffle=False)
        batches = _collect_all_batches(sampler)
        assert len(batches) == 3
        for batch in batches:
            assert len(batch) == 3


# ============================================================================
# drop_last
# ============================================================================


class TestDropLast:
    __test__ = True

    def test_drop_last_removes_incomplete_batches(self) -> None:
        ts = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]).float()
        sampler = TimestampGroupedBatchSampler(ts, batch_size=3, shuffle=False, drop_last=True)
        batches = _collect_all_batches(sampler)
        for batch in batches:
            assert len(batch) == 3
        assert len(batches) == 2

    def test_drop_last_false_keeps_incomplete(self) -> None:
        ts = torch.tensor([0, 0, 0, 0, 0]).float()
        sampler = TimestampGroupedBatchSampler(ts, batch_size=3, shuffle=False, drop_last=False)
        batches = _collect_all_batches(sampler)
        assert len(batches) == 2
        assert len(batches[-1]) == 2


# ============================================================================
# __len__
# ============================================================================


class TestLen:
    __test__ = True

    def test_len_matches_actual_batches(self, timestamps_10_uniform: Tensor) -> None:
        sampler = TimestampGroupedBatchSampler(timestamps_10_uniform, batch_size=8, shuffle=False)
        assert len(sampler) == len(_collect_all_batches(sampler))

    def test_len_with_drop_last(self, timestamps_10_uniform: Tensor) -> None:
        sampler = TimestampGroupedBatchSampler(
            timestamps_10_uniform, batch_size=8, shuffle=False, drop_last=True
        )
        assert len(sampler) == len(_collect_all_batches(sampler))

    def test_len_single_timestamp(self, timestamps_single: Tensor) -> None:
        sampler = TimestampGroupedBatchSampler(timestamps_single, batch_size=16, shuffle=False)
        assert len(sampler) == len(_collect_all_batches(sampler))


# ============================================================================
# Shuffle behaviour
# ============================================================================


class TestShuffleBehaviour:
    __test__ = True

    def test_shuffle_changes_batch_order_across_epochs(self, timestamps_10_uniform: Tensor) -> None:
        """Two epochs with shuffle=True should (almost surely) differ."""
        sampler = TimestampGroupedBatchSampler(timestamps_10_uniform, batch_size=4, shuffle=True)
        epoch1 = _flatten_batches(_collect_all_batches(sampler))
        epoch2 = _flatten_batches(_collect_all_batches(sampler))
        assert epoch1 != epoch2, "Two shuffled epochs should differ"

    def test_no_shuffle_is_deterministic(self, timestamps_10_uniform: Tensor) -> None:
        sampler = TimestampGroupedBatchSampler(timestamps_10_uniform, batch_size=4, shuffle=False)
        epoch1 = _flatten_batches(_collect_all_batches(sampler))
        epoch2 = _flatten_batches(_collect_all_batches(sampler))
        assert epoch1 == epoch2

    def test_generator_for_reproducibility(self, timestamps_10_uniform: Tensor) -> None:
        g1 = torch.Generator().manual_seed(99)
        g2 = torch.Generator().manual_seed(99)
        s1 = TimestampGroupedBatchSampler(
            timestamps_10_uniform, batch_size=4, shuffle=True, generator=g1
        )
        s2 = TimestampGroupedBatchSampler(
            timestamps_10_uniform, batch_size=4, shuffle=True, generator=g2
        )
        b1 = _flatten_batches(_collect_all_batches(s1))
        b2 = _flatten_batches(_collect_all_batches(s2))
        assert b1 == b2


# ============================================================================
# Edge cases
# ============================================================================


class TestEdgeCases:
    __test__ = True

    def test_single_element(self) -> None:
        ts = torch.tensor([7.0])
        sampler = TimestampGroupedBatchSampler(ts, batch_size=4, shuffle=False)
        batches = _collect_all_batches(sampler)
        assert batches == [[0]]

    def test_two_groups_one_element_each(self) -> None:
        ts = torch.tensor([1.0, 2.0])
        sampler = TimestampGroupedBatchSampler(ts, batch_size=4, shuffle=False)
        batches = _collect_all_batches(sampler)
        assert len(batches) == 2
        flat = _flatten_batches(batches)
        assert sorted(flat) == [0, 1]

    def test_empty_after_drop_last(self) -> None:
        """If every group has fewer elements than batch_size, drop_last yields nothing."""
        ts = torch.tensor([0.0, 1.0, 2.0])
        sampler = TimestampGroupedBatchSampler(ts, batch_size=4, shuffle=False, drop_last=True)
        batches = _collect_all_batches(sampler)
        assert len(batches) == 0
        assert len(sampler) == 0


# ============================================================================
# Distributed (DDP) sharding — tested without real distributed processes
# ============================================================================


class TestDistributedSharding:
    __test__ = True

    @pytest.fixture()
    def ts_120(self) -> Tensor:
        """120 seed edges across 6 timestamps (20 each)."""
        return torch.arange(6).repeat_interleave(20).float()

    def test_all_ranks_cover_all_indices(self, ts_120: Tensor) -> None:
        """Union of batches across all ranks covers every index exactly once (modulo padding)."""
        num_replicas = 4
        all_indices: list[int] = []
        for rank in range(num_replicas):
            sampler = TimestampGroupedBatchSampler(
                ts_120, batch_size=8, shuffle=False, num_replicas=num_replicas, rank=rank
            )
            all_indices.extend(_flatten_batches(_collect_all_batches(sampler)))
        unique = set(all_indices)
        assert unique == set(range(len(ts_120)))

    def test_ranks_yield_equal_batch_counts(self, ts_120: Tensor) -> None:
        num_replicas = 3
        counts = []
        for rank in range(num_replicas):
            sampler = TimestampGroupedBatchSampler(
                ts_120, batch_size=8, shuffle=False, num_replicas=num_replicas, rank=rank
            )
            counts.append(len(_collect_all_batches(sampler)))
        assert len(set(counts)) == 1, f"Rank batch counts differ: {counts}"

    def test_len_matches_actual_distributed(self, ts_120: Tensor) -> None:
        num_replicas = 4
        for rank in range(num_replicas):
            sampler = TimestampGroupedBatchSampler(
                ts_120, batch_size=8, shuffle=False, num_replicas=num_replicas, rank=rank
            )
            assert len(sampler) == len(_collect_all_batches(sampler))

    def test_ranks_are_disjoint_no_shuffle(self, ts_120: Tensor) -> None:
        """Without shuffle, each batch appears on exactly one rank (except padding)."""
        num_replicas = 4
        rank_batches: list[list[list[int]]] = []
        for rank in range(num_replicas):
            sampler = TimestampGroupedBatchSampler(
                ts_120, batch_size=8, shuffle=False, num_replicas=num_replicas, rank=rank
            )
            rank_batches.append(_collect_all_batches(sampler))

        total = sum(len(b) for b in rank_batches)
        per_rank = total // num_replicas
        for rb in rank_batches:
            assert len(rb) == per_rank

    def test_timestamp_homogeneity_preserved_in_ddp(self, ts_120: Tensor) -> None:
        num_replicas = 4
        for rank in range(num_replicas):
            sampler = TimestampGroupedBatchSampler(
                ts_120, batch_size=8, shuffle=True, num_replicas=num_replicas, rank=rank
            )
            for batch in sampler:
                ts_in_batch = ts_120[batch]
                assert (ts_in_batch == ts_in_batch[0]).all()

    def test_shuffle_deterministic_across_ranks(self, ts_120: Tensor) -> None:
        """All ranks use the same epoch-based seed, so the global batch order is shared."""
        num_replicas = 2
        for epoch in range(3):
            batches_per_rank: list[list[list[int]]] = []
            for rank in range(num_replicas):
                sampler = TimestampGroupedBatchSampler(
                    ts_120, batch_size=8, shuffle=True, num_replicas=num_replicas, rank=rank
                )
                sampler.set_epoch(epoch)
                batches_per_rank.append(_collect_all_batches(sampler))
            assert len(batches_per_rank[0]) == len(batches_per_rank[1])

    def test_set_epoch_changes_shuffle_order(self, ts_120: Tensor) -> None:
        num_replicas = 2
        sampler = TimestampGroupedBatchSampler(
            ts_120, batch_size=8, shuffle=True, num_replicas=num_replicas, rank=0
        )
        sampler.set_epoch(0)
        epoch0 = _flatten_batches(_collect_all_batches(sampler))
        sampler.set_epoch(1)
        epoch1 = _flatten_batches(_collect_all_batches(sampler))
        assert epoch0 != epoch1, "Different epochs should produce different orderings"

    def test_single_replica_matches_non_distributed(self, timestamps_10_uniform: Tensor) -> None:
        """num_replicas=1, rank=0 behaves identically to the default constructor."""
        s_default = TimestampGroupedBatchSampler(timestamps_10_uniform, batch_size=8, shuffle=False)
        s_explicit = TimestampGroupedBatchSampler(
            timestamps_10_uniform, batch_size=8, shuffle=False, num_replicas=1, rank=0
        )
        assert _collect_all_batches(s_default) == _collect_all_batches(s_explicit)
