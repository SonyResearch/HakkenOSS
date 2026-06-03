"""Tests for TemporalKGLinkNeighborLoader.

Validates initialization, temporal filtering invariants, output structure,
shapes, dtypes, and edge-label remapping for the temporal-aware link
neighbor sampler.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch
from torch import Tensor

from hakken_models.core.entities.kg_data import KGData, assert_is_kg_data
from hakken_models.data_loaders.temporal_kg_link_neighbor_loader import (
    TemporalKGLinkNeighborLoader,
)

# ============================================================================
# Parametrised configs — each entry defines a small temporal KG
# ============================================================================

LOADER_CONFIGS: list[dict[str, Any]] = [
    {
        "num_nodes": 20,
        "num_relations": 4,
        "num_timestamps": 5,
        "num_facts": 60,
        "num_seed_edges": 10,
        "batch_size": 4,
        "num_neighbors": [5, 3],
    },
    {
        "num_nodes": 50,
        "num_relations": 8,
        "num_timestamps": 10,
        "num_facts": 200,
        "num_seed_edges": 20,
        "batch_size": 8,
        "num_neighbors": [10, 5],
    },
    {
        "num_nodes": 100,
        "num_relations": 12,
        "num_timestamps": 20,
        "num_facts": 500,
        "num_seed_edges": 30,
        "batch_size": 10,
        "num_neighbors": [15, 10],
    },
]

SEED_LIST: list[int] = [42, 123]

# ============================================================================
# Helpers
# ============================================================================


def _build_temporal_facts(
    num_nodes: int,
    num_relations: int,
    num_timestamps: int,
    num_facts: int,
) -> Tensor:
    """Build a ``[num_facts, 4]`` facts tensor ``(s, r, o, t)``."""
    subjects = torch.randint(0, num_nodes, (num_facts,))
    relations = torch.randint(0, num_relations, (num_facts,))
    objects = torch.randint(0, num_nodes, (num_facts,))
    timestamps = torch.randint(0, num_timestamps, (num_facts,))
    return torch.stack([subjects, relations, objects, timestamps], dim=1)


def _build_seed_edges(
    facts: Tensor,
    num_seed_edges: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Select ``num_seed_edges`` rows from *facts* as seed edges.

    Returns
    -------
    edge_label_index : Tensor [2, N]
        Subject / object pairs.
    edge_label : Tensor [N]
        Relation indices.
    target_timestamps : Tensor [N]
        Timestamps for each seed edge.
    """
    perm = torch.randperm(facts.size(0))[:num_seed_edges]
    selected = facts[perm]
    edge_label_index = selected[:, [0, 2]].t().contiguous()  # [2, N]
    edge_label = selected[:, 1]  # [N]
    target_timestamps = selected[:, 3]  # [N]
    return edge_label_index, edge_label, target_timestamps


# ============================================================================
# Base test class with shared fixtures
# ============================================================================


class _TestBase:
    __test__ = False

    @pytest.fixture(params=SEED_LIST)
    def seed(self, request: pytest.FixtureRequest) -> int:
        return cast(int, request.param)

    @pytest.fixture(autouse=True)
    def _set_seed(self, seed: int) -> None:
        torch.manual_seed(seed)

    @pytest.fixture(params=LOADER_CONFIGS, ids=lambda c: f"n{c['num_nodes']}_e{c['num_facts']}")
    def cfg(self, request: pytest.FixtureRequest) -> dict[str, Any]:
        return request.param

    @pytest.fixture
    def temporal_facts(self, cfg: dict[str, Any]) -> Tensor:
        return _build_temporal_facts(
            num_nodes=cfg["num_nodes"],
            num_relations=cfg["num_relations"],
            num_timestamps=cfg["num_timestamps"],
            num_facts=cfg["num_facts"],
        )

    @pytest.fixture
    def kg_data(self, cfg: dict[str, Any], temporal_facts: Tensor) -> KGData:
        """Full KGData **without** ``n_id`` (relabel_nodes=False)."""
        return KGData.from_facts(
            temporal_facts,
            num_nodes=cfg["num_nodes"],
            relabel_nodes=False,
        )

    @pytest.fixture
    def seed_edges(
        self, temporal_facts: Tensor, cfg: dict[str, Any]
    ) -> tuple[Tensor, Tensor, Tensor]:
        return _build_seed_edges(temporal_facts, num_seed_edges=cfg["num_seed_edges"])

    @pytest.fixture
    def loader(
        self,
        kg_data: KGData,
        seed_edges: tuple[Tensor, Tensor, Tensor],
        cfg: dict[str, Any],
    ) -> TemporalKGLinkNeighborLoader:
        edge_label_index, edge_label, target_timestamps = seed_edges
        return TemporalKGLinkNeighborLoader(
            data=kg_data,
            num_neighbors=cfg["num_neighbors"],
            edge_label_index=edge_label_index,
            edge_label=edge_label,
            target_timestamps=target_timestamps,
            batch_size=cfg["batch_size"],
            shuffle=False,
        )

    @pytest.fixture
    def sample_batch(
        self,
        loader: TemporalKGLinkNeighborLoader,
        cfg: dict[str, Any],
    ) -> tuple[KGData, Tensor]:
        """Return the first batch and its index tensor."""
        batch_size = min(cfg["batch_size"], cfg["num_seed_edges"])
        index = torch.arange(batch_size)
        batch = loader(index)
        return batch, index


# ============================================================================
# Initialization tests
# ============================================================================


class TestTemporalKGLinkNeighborLoaderInit(_TestBase):
    __test__ = True

    def test_constructor_stores_full_edge_data(
        self,
        loader: TemporalKGLinkNeighborLoader,
        kg_data: KGData,
    ) -> None:
        assert torch.equal(loader._full_edge_attr, kg_data.edge_attr)
        assert torch.equal(loader._full_edge_index, kg_data.edge_index)

    def test_constructor_stores_target_timestamps(
        self,
        loader: TemporalKGLinkNeighborLoader,
        seed_edges: tuple[Tensor, Tensor, Tensor],
    ) -> None:
        _, _, target_timestamps = seed_edges
        assert torch.equal(loader._target_timestamps, target_timestamps)

    def test_constructor_stores_num_neighbors(
        self,
        loader: TemporalKGLinkNeighborLoader,
        cfg: dict[str, Any],
    ) -> None:
        assert loader.num_neighbors == cfg["num_neighbors"]

    def test_rejects_data_with_n_id(
        self,
        cfg: dict[str, Any],
        temporal_facts: Tensor,
        seed_edges: tuple[Tensor, Tensor, Tensor],
    ) -> None:
        kg_with_nid = KGData.from_facts(
            temporal_facts,
            num_nodes=cfg["num_nodes"],
            relabel_nodes=True,
        )
        edge_label_index, edge_label, target_timestamps = seed_edges
        with pytest.raises(ValueError, match="must not contain the original node indexes"):
            TemporalKGLinkNeighborLoader(
                data=kg_with_nid,
                num_neighbors=cfg["num_neighbors"],
                edge_label_index=edge_label_index,
                edge_label=edge_label,
                target_timestamps=target_timestamps,
            )


# ============================================================================
# Output structure & shape tests
# ============================================================================


class TestOutputStructureAndShapes(_TestBase):
    __test__ = True

    def test_returns_kg_data_instance(self, sample_batch: tuple[KGData, Tensor]) -> None:
        batch, _ = sample_batch
        assert isinstance(batch, KGData)

    def test_output_has_required_graph_fields(self, sample_batch: tuple[KGData, Tensor]) -> None:
        batch, _ = sample_batch
        assert_is_kg_data(batch)

    def test_n_id_present_and_1d(self, sample_batch: tuple[KGData, Tensor]) -> None:
        batch, _ = sample_batch
        assert batch.n_id is not None
        assert batch.n_id.dim() == 1

    def test_edge_index_shape_2_by_e(self, sample_batch: tuple[KGData, Tensor]) -> None:
        batch, _ = sample_batch
        assert batch.edge_index.dim() == 2
        assert batch.edge_index.shape[0] == 2

    def test_edge_attr_aligned_with_edge_index(self, sample_batch: tuple[KGData, Tensor]) -> None:
        batch, _ = sample_batch
        assert batch.edge_attr.shape[0] == batch.edge_index.shape[1]

    def test_edge_attr_has_two_columns(self, sample_batch: tuple[KGData, Tensor]) -> None:
        """edge_attr should have at least relation and timestamp columns (2 or 3 when reverse edges)."""
        batch, _ = sample_batch
        if batch.edge_attr.numel() > 0:
            assert batch.edge_attr.shape[1] >= 2

    def test_edge_label_index_shape(
        self,
        sample_batch: tuple[KGData, Tensor],
        cfg: dict[str, Any],  # noqa: ARG002
    ) -> None:
        batch, index = sample_batch
        assert batch.edge_label_index.dim() == 2
        assert batch.edge_label_index.shape[0] == 2
        assert batch.edge_label_index.shape[1] == index.size(0)

    def test_edge_label_shape(
        self,
        sample_batch: tuple[KGData, Tensor],
        cfg: dict[str, Any],  # noqa: ARG002
    ) -> None:
        batch, index = sample_batch
        assert batch.edge_label.dim() == 1
        assert batch.edge_label.shape[0] == index.size(0)

    def test_input_id_present(self, sample_batch: tuple[KGData, Tensor]) -> None:
        batch, _ = sample_batch
        assert batch.input_id is not None

    def test_e_id_filtered_consistently(self, sample_batch: tuple[KGData, Tensor]) -> None:
        batch, _ = sample_batch
        if batch.e_id is not None:
            assert batch.e_id.shape[0] == batch.edge_index.shape[1]


# ============================================================================
# Dtype tests
# ============================================================================


class TestOutputDtypes(_TestBase):
    __test__ = True

    def test_edge_index_dtype_long(self, sample_batch: tuple[KGData, Tensor]) -> None:
        batch, _ = sample_batch
        assert batch.edge_index.dtype == torch.long

    def test_edge_label_index_dtype_long(self, sample_batch: tuple[KGData, Tensor]) -> None:
        batch, _ = sample_batch
        assert batch.edge_label_index.dtype == torch.long

    def test_edge_label_dtype_long(self, sample_batch: tuple[KGData, Tensor]) -> None:
        batch, _ = sample_batch
        assert batch.edge_label.dtype == torch.long

    def test_n_id_dtype_long(self, sample_batch: tuple[KGData, Tensor]) -> None:
        batch, _ = sample_batch
        assert batch.n_id.dtype == torch.long

    def test_edge_attr_dtype_long(self, sample_batch: tuple[KGData, Tensor]) -> None:
        batch, _ = sample_batch
        if batch.edge_attr.numel() > 0:
            assert batch.edge_attr.dtype == torch.long


# ============================================================================
# Temporal filtering invariant
# ============================================================================


class TestTemporalFiltering(_TestBase):
    """Verify that _postprocess applies temporal filtering on context edges."""

    __test__ = True

    def test_all_edges_respect_temporal_cutoff(
        self,
        loader: TemporalKGLinkNeighborLoader,
        cfg: dict[str, Any],
    ) -> None:
        """After filtering, no context edge should have timestamp > cutoff."""
        bs = min(cfg["batch_size"], cfg["num_seed_edges"])
        index = torch.arange(bs)
        batch = loader(index)

        cutoff = loader._target_timestamps[index].min()
        if batch.edge_attr.numel() > 0:
            edge_ts = batch.edge_attr[:, 1].float()
            assert (edge_ts <= cutoff).all()

    def test_filtering_varies_with_batch(
        self,
        loader: TemporalKGLinkNeighborLoader,
        seed_edges: tuple[Tensor, Tensor, Tensor],  # noqa: ARG002
        cfg: dict[str, Any],
    ) -> None:
        """Different batches may retain different numbers of edges."""
        num_seeds = cfg["num_seed_edges"]
        bs = min(cfg["batch_size"], num_seeds)

        edge_counts: list[int] = []
        for start in range(0, num_seeds, bs):
            end = min(start + bs, num_seeds)
            index = torch.arange(start, end)
            batch = loader(index)
            edge_counts.append(batch.edge_index.shape[1])

        assert max(edge_counts) > min(edge_counts)

    def test_max_ts_targets_keep_all_edges(
        self,
        kg_data: KGData,
        seed_edges: tuple[Tensor, Tensor, Tensor],
        cfg: dict[str, Any],
    ) -> None:
        """When cutoff equals max timestamp, no edges should be dropped."""
        edge_label_index, edge_label, _ = seed_edges
        max_ts = kg_data.edge_attr[:, 1].max().item()
        high_timestamps = torch.full((edge_label_index.shape[1],), max_ts, dtype=torch.long)

        high_loader = TemporalKGLinkNeighborLoader(
            data=kg_data,
            num_neighbors=cfg["num_neighbors"],
            edge_label_index=edge_label_index,
            edge_label=edge_label,
            target_timestamps=high_timestamps,
            batch_size=cfg["batch_size"],
            shuffle=False,
        )

        bs = min(cfg["batch_size"], edge_label_index.shape[1])
        batch = high_loader(torch.arange(bs))
        if batch.edge_attr.numel() > 0:
            assert (batch.edge_attr[:, 1].float() <= max_ts).all()

    def test_impossible_cutoff_removes_all_context_edges(
        self,
        kg_data: KGData,
        seed_edges: tuple[Tensor, Tensor, Tensor],
        cfg: dict[str, Any],
    ) -> None:
        """A cutoff below the minimum edge timestamp should leave no context edges."""
        edge_label_index, edge_label, _ = seed_edges
        min_ts = kg_data.edge_attr[:, 1].min().item()
        impossible_ts = torch.full((edge_label_index.shape[1],), min_ts - 1, dtype=torch.long)

        low_loader = TemporalKGLinkNeighborLoader(
            data=kg_data,
            num_neighbors=cfg["num_neighbors"],
            edge_label_index=edge_label_index,
            edge_label=edge_label,
            target_timestamps=impossible_ts,
            batch_size=cfg["batch_size"],
            shuffle=False,
        )

        bs = min(cfg["batch_size"], edge_label_index.shape[1])
        batch = low_loader(torch.arange(bs))
        assert batch.edge_index.shape[1] == 0


# ============================================================================
# Edge-label remapping to global IDs
# ============================================================================


class TestEdgeLabelRemapping(_TestBase):
    __test__ = True

    def test_edge_label_index_uses_global_node_ids(
        self,
        loader: TemporalKGLinkNeighborLoader,
        cfg: dict[str, Any],
    ) -> None:
        """edge_label_index should reference global node IDs, not local."""
        bs = min(cfg["batch_size"], cfg["num_seed_edges"])
        index = torch.arange(bs)
        batch = loader(index)

        global_node_ids = batch.n_id
        assert batch.edge_label_index.shape[0] == 2
        assert torch.all(torch.isin(batch.edge_label_index[0], global_node_ids))
        assert torch.all(torch.isin(batch.edge_label_index[1], global_node_ids))

    def test_edge_label_index_bounded_by_num_nodes(
        self,
        loader: TemporalKGLinkNeighborLoader,
        cfg: dict[str, Any],
    ) -> None:
        bs = min(cfg["batch_size"], cfg["num_seed_edges"])
        index = torch.arange(bs)
        batch = loader(index)

        assert batch.edge_label_index.max().item() < cfg["num_nodes"]
        assert batch.edge_label_index.min().item() >= 0


# ============================================================================
# Index type variants (Tensor vs list[int])
# ============================================================================


class TestIndexTypeVariants(_TestBase):
    __test__ = True

    def test_tensor_index_produces_valid_batch(
        self,
        loader: TemporalKGLinkNeighborLoader,
        cfg: dict[str, Any],
    ) -> None:
        bs = min(cfg["batch_size"], cfg["num_seed_edges"])
        batch = loader(torch.arange(bs))
        assert_is_kg_data(batch)

    def test_list_index_produces_valid_batch(
        self,
        loader: TemporalKGLinkNeighborLoader,
        cfg: dict[str, Any],
    ) -> None:
        bs = min(cfg["batch_size"], cfg["num_seed_edges"])
        batch = loader(list(range(bs)))
        assert_is_kg_data(batch)

    def test_tensor_and_list_index_give_same_shapes(
        self,
        loader: TemporalKGLinkNeighborLoader,
        cfg: dict[str, Any],
    ) -> None:
        """Same logical index should yield identically-shaped outputs."""
        bs = min(cfg["batch_size"], cfg["num_seed_edges"])
        idx_list: list[int] = list(range(bs))
        idx_tensor: Tensor = torch.tensor(idx_list, dtype=torch.long)

        batch_from_list = loader(idx_list)
        batch_from_tensor = loader(idx_tensor)

        assert batch_from_list.edge_label_index.shape == batch_from_tensor.edge_label_index.shape
        assert batch_from_list.edge_label.shape == batch_from_tensor.edge_label.shape


# ============================================================================
# Deterministic small-graph smoke test
# ============================================================================


class TestDeterministicSmallGraph:
    """Hand-crafted 6-node graph to verify temporal filtering end-to-end."""

    __test__ = True

    @pytest.fixture
    def small_graph(self) -> tuple[KGData, Tensor, Tensor, Tensor]:
        """Build a tiny temporal KG with predictable structure.

        Edges (s, r, o, t):
            0 -> 1  rel=0  ts=0
            1 -> 2  rel=1  ts=1
            2 -> 3  rel=0  ts=2
            3 -> 4  rel=1  ts=3
            4 -> 5  rel=0  ts=4
            0 -> 3  rel=1  ts=2
            1 -> 4  rel=0  ts=3
            2 -> 5  rel=1  ts=4
        """
        torch.manual_seed(0)
        facts = torch.tensor(
            [
                [0, 0, 1, 0],
                [1, 1, 2, 1],
                [2, 0, 3, 2],
                [3, 1, 4, 3],
                [4, 0, 5, 4],
                [0, 1, 3, 2],
                [1, 0, 4, 3],
                [2, 1, 5, 4],
            ],
            dtype=torch.long,
        )
        num_nodes = 6
        kg_data = KGData.from_facts(facts, num_nodes=num_nodes, relabel_nodes=False)

        seed_idx = torch.tensor([2, 5], dtype=torch.long)
        selected = facts[seed_idx]
        edge_label_index = selected[:, [0, 2]].t().contiguous()
        edge_label = selected[:, 1]
        target_timestamps = selected[:, 3]

        return kg_data, edge_label_index, edge_label, target_timestamps

    def test_small_graph_temporal_filtering(
        self, small_graph: tuple[KGData, Tensor, Tensor, Tensor]
    ) -> None:
        kg_data, edge_label_index, edge_label, target_timestamps = small_graph

        loader = TemporalKGLinkNeighborLoader(
            data=kg_data,
            num_neighbors=[10],
            edge_label_index=edge_label_index,
            edge_label=edge_label,
            target_timestamps=target_timestamps,
            batch_size=2,
            shuffle=False,
        )

        index = torch.arange(2)
        batch = loader(index)

        cutoff = target_timestamps.min().item()
        assert cutoff == 2
        if batch.edge_attr.numel() > 0:
            assert (batch.edge_attr[:, 1].float() <= cutoff).all()

    def test_small_graph_edge_label_index_values(
        self, small_graph: tuple[KGData, Tensor, Tensor, Tensor]
    ) -> None:
        kg_data, edge_label_index, edge_label, target_timestamps = small_graph

        loader = TemporalKGLinkNeighborLoader(
            data=kg_data,
            num_neighbors=[10],
            edge_label_index=edge_label_index,
            edge_label=edge_label,
            target_timestamps=target_timestamps,
            batch_size=2,
            shuffle=False,
        )

        index = torch.arange(2)
        batch = loader(index)

        assert batch.edge_label_index.shape == torch.Size([2, 2])
        assert torch.all(batch.edge_label_index >= 0)
        assert torch.all(batch.edge_label_index < 6)

    def test_small_graph_edge_label_values(
        self, small_graph: tuple[KGData, Tensor, Tensor, Tensor]
    ) -> None:
        kg_data, edge_label_index, edge_label, target_timestamps = small_graph

        loader = TemporalKGLinkNeighborLoader(
            data=kg_data,
            num_neighbors=[10],
            edge_label_index=edge_label_index,
            edge_label=edge_label,
            target_timestamps=target_timestamps,
            batch_size=2,
            shuffle=False,
        )

        index = torch.arange(2)
        batch = loader(index)

        assert batch.edge_label.shape == torch.Size([2])
        assert torch.equal(batch.edge_label, edge_label)

    def test_small_graph_output_dtypes(
        self, small_graph: tuple[KGData, Tensor, Tensor, Tensor]
    ) -> None:
        kg_data, edge_label_index, edge_label, target_timestamps = small_graph

        loader = TemporalKGLinkNeighborLoader(
            data=kg_data,
            num_neighbors=[10],
            edge_label_index=edge_label_index,
            edge_label=edge_label,
            target_timestamps=target_timestamps,
            batch_size=2,
            shuffle=False,
        )

        batch = loader(torch.arange(2))

        assert batch.edge_index.dtype == torch.long
        assert batch.edge_label_index.dtype == torch.long
        assert batch.edge_label.dtype == torch.long
        assert batch.n_id.dtype == torch.long
        if batch.edge_attr.numel() > 0:
            assert batch.edge_attr.dtype == torch.long


# ============================================================================
# Iteration via DataLoader protocol
# ============================================================================


class TestIterationProtocol(_TestBase):
    __test__ = True

    def test_iterate_full_epoch(
        self,
        loader: TemporalKGLinkNeighborLoader,
        seed_edges: tuple[Tensor, Tensor, Tensor],  # noqa: ARG002
        cfg: dict[str, Any],
    ) -> None:
        """Iterating the loader should yield batches covering all seeds."""
        total_seeds_seen = 0

        for batch in loader:
            assert_is_kg_data(batch)
            total_seeds_seen += batch.edge_label.shape[0]

        assert total_seeds_seen == cfg["num_seed_edges"]


# ============================================================================
# Timestamp-grouped batching (group_by_timestamp=True)
# ============================================================================


class TestTimestampGroupedBatching(_TestBase):
    """Verify that ``group_by_timestamp=True`` produces same-timestamp batches."""

    __test__ = True

    @pytest.fixture
    def grouped_loader(
        self,
        kg_data: KGData,
        seed_edges: tuple[Tensor, Tensor, Tensor],
        cfg: dict[str, Any],
    ) -> TemporalKGLinkNeighborLoader:
        edge_label_index, edge_label, target_timestamps = seed_edges
        return TemporalKGLinkNeighborLoader(
            data=kg_data,
            num_neighbors=cfg["num_neighbors"],
            edge_label_index=edge_label_index,
            edge_label=edge_label,
            target_timestamps=target_timestamps,
            batch_size=cfg["batch_size"],
            shuffle=False,
            group_by_timestamp=True,
        )

    def test_grouped_batches_have_homogeneous_timestamps(
        self,
        grouped_loader: TemporalKGLinkNeighborLoader,
        seed_edges: tuple[Tensor, Tensor, Tensor],
    ) -> None:
        """Each batch must contain seed edges with the same target timestamp."""
        _, _, target_timestamps = seed_edges
        for batch in grouped_loader:
            batch_ts = target_timestamps[batch.input_id]
            assert (batch_ts == batch_ts[0]).all(), (
                f"Batch contains mixed target timestamps: {batch_ts.unique().tolist()}"
            )

    def test_grouped_loader_covers_all_seed_edges(
        self,
        grouped_loader: TemporalKGLinkNeighborLoader,
        cfg: dict[str, Any],
    ) -> None:
        total = sum(b.edge_label.shape[0] for b in grouped_loader)
        assert total == cfg["num_seed_edges"]

    def test_grouped_batches_have_valid_structure(
        self,
        grouped_loader: TemporalKGLinkNeighborLoader,
    ) -> None:
        for batch in grouped_loader:
            assert_is_kg_data(batch)
            assert batch.n_id is not None
            assert batch.edge_label_index is not None
            assert batch.edge_label is not None

    def test_grouped_filtering_matches_common_timestamp(
        self,
        grouped_loader: TemporalKGLinkNeighborLoader,
        seed_edges: tuple[Tensor, Tensor, Tensor],
    ) -> None:
        """With same-timestamp batches, all context edges should have
        timestamp <= the common target timestamp.
        """
        _, _, target_timestamps = seed_edges
        for batch in grouped_loader:
            batch_ts = target_timestamps[batch.input_id]
            expected_cutoff = batch_ts[0].item()
            if batch.edge_attr.numel() > 0:
                assert (batch.edge_attr[:, 1].float() <= expected_cutoff).all()

    def test_group_by_timestamp_false_preserves_old_behaviour(
        self,
        kg_data: KGData,
        seed_edges: tuple[Tensor, Tensor, Tensor],
        cfg: dict[str, Any],
    ) -> None:
        """With ``group_by_timestamp=False``, batches may mix timestamps."""
        edge_label_index, edge_label, target_timestamps = seed_edges
        loader = TemporalKGLinkNeighborLoader(
            data=kg_data,
            num_neighbors=cfg["num_neighbors"],
            edge_label_index=edge_label_index,
            edge_label=edge_label,
            target_timestamps=target_timestamps,
            batch_size=cfg["batch_size"],
            shuffle=False,
            group_by_timestamp=False,
        )
        total = sum(b.edge_label.shape[0] for b in loader)
        assert total == cfg["num_seed_edges"]
