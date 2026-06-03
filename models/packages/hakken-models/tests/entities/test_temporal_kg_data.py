from typing import Any, cast

import polars as pl
import pytest
import torch
from torch import Tensor

from hakken_models.core.entities.kg_data import KGData, has_attr_without_none
from hakken_models.core.entities.temporal_kg_data import TemporalKGData

# ============================================================================
# Parameter Sets for KGData Constructor
# ============================================================================

KG_DATA_CONFIGS = [
    {"num_nodes": 10, "num_relations": 5, "num_timestamps": 3, "num_facts": 15, "num_domains": 5},
    {
        "num_nodes": 100,
        "num_relations": 20,
        "num_timestamps": 1,
        "num_facts": 500,
        "num_domains": 2,
    },
    {
        "num_nodes": 10_000,
        "num_relations": 100,
        "num_timestamps": 80,
        "num_facts": 100_000,
        "num_domains": 10,
    },
    {"num_nodes": 50, "num_relations": 10, "num_facts": 1000, "num_domains": 4},
]

SEED_LIST = [10, 52, 101]

# ============================================================================
# Base Test class
# ============================================================================


class TestBaseKGData:
    __test__ = True

    @pytest.fixture(params=SEED_LIST)
    def seed(self, request: pytest.FixtureRequest) -> int:
        """Parametrized seed fixture."""
        return request.param

    @pytest.fixture(autouse=True)
    def set_seed(self, seed: int):
        """Automatically set random seed before each test for reproducibility."""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        yield

    @pytest.fixture(params=KG_DATA_CONFIGS)
    def kg_config(self, request: pytest.FixtureRequest) -> dict[str, Any]:
        """Parametrized configuration fixture."""
        return request.param

    # ========================================================================
    # Test Data Fixtures (depend on kg_config)
    # ========================================================================

    @pytest.fixture
    def sample_facts(self, kg_config: dict, seed: int) -> Tensor:
        """Sample facts tensor for testing, respecting node/relation bounds."""
        assert seed is not None
        num_nodes = cast(int, kg_config.get("num_nodes"))
        num_relations = cast(int, kg_config.get("num_relations"))
        num_facts = cast(int, kg_config.get("num_facts"))
        subjects = torch.randint(0, num_nodes, (num_facts,))
        relations = torch.randint(0, num_relations, (num_facts,))
        objects = torch.randint(0, num_nodes, (num_facts,))
        if "num_timestamps" in kg_config:
            num_timestamps = cast(int, kg_config.get("num_timestamps"))
            timestamps = torch.randint(0, num_timestamps, (num_facts,))
            return torch.stack([subjects, relations, objects, timestamps], dim=1)
        return torch.stack([subjects, relations, objects], dim=1)

    @pytest.fixture
    def sample_temporal_facts(self, sample_facts: Tensor) -> Tensor:
        """Sample temporal facts tensor - only returns facts with timestamps."""
        # Only return if this config has timestamps (4 columns)
        if sample_facts.shape[1] == 4:
            return sample_facts
        pytest.skip("Config does not include num_timestamps, skipping temporal tests")

    @pytest.fixture
    def sample_domains_mapping_df(self, kg_config: dict, sample_facts: Tensor) -> pl.DataFrame:
        """Create a domains mapping DataFrame for the nodes in sample_facts."""
        # Extract unique node IDs from facts
        edge_pairs = sample_facts[:, [0, 2]]
        unique_node_ids = edge_pairs.unique().tolist()
        num_domains = kg_config.get("num_domains")
        domain_ids = torch.randint(0, num_domains, (len(unique_node_ids),)).tolist()
        return pl.DataFrame(
            {
                "node_id": unique_node_ids,
                "domain_id": domain_ids,
            }
        )

    @pytest.fixture
    def sample_kg_data(self, kg_config: dict, sample_temporal_facts: Tensor) -> KGData:
        """Create a KGData instance from temporal facts for TemporalKGData tests."""
        num_nodes = kg_config.get("num_nodes")
        return KGData.from_facts(sample_temporal_facts, num_nodes=num_nodes, relabel_nodes=True)

    @pytest.fixture
    def temporal_kg_data(self, sample_kg_data: KGData) -> TemporalKGData:
        """Create a TemporalKGData instance for testing."""
        return TemporalKGData(sample_kg_data, edge_attr_timestamp_col=1)


class TestTemporalKGDataInitialization(TestBaseKGData):
    """Test class for TemporalKGData initialization and basic properties."""

    __test__ = True

    def test_temporal_kg_data_initialization(self, sample_kg_data: KGData) -> None:
        """Test that TemporalKGData initializes correctly."""
        temporal_kg = TemporalKGData(sample_kg_data, edge_attr_timestamp_col=1)

        assert temporal_kg.data is sample_kg_data
        assert temporal_kg.edge_timestamps is not None
        assert temporal_kg.unique_timestamps is not None
        assert len(temporal_kg.unique_timestamps) > 0

    def test_temporal_kg_data_edge_timestamps_correctness(
        self, sample_kg_data: KGData, sample_temporal_facts: Tensor
    ) -> None:
        """Test that edge_timestamps correctly extracts timestamp column."""
        temporal_kg = TemporalKGData(sample_kg_data, edge_attr_timestamp_col=1)

        # edge_attr column 1 should contain timestamps
        expected_timestamps = sample_temporal_facts[:, 3]
        assert torch.equal(temporal_kg.edge_timestamps, expected_timestamps)

    def test_temporal_kg_data_unique_timestamps(self, sample_kg_data: KGData) -> None:
        """Test that unique_timestamps contains all unique timestamp values."""
        temporal_kg = TemporalKGData(sample_kg_data, edge_attr_timestamp_col=1)

        expected_unique = torch.unique(temporal_kg.edge_timestamps).tolist()
        assert set(temporal_kg.unique_timestamps) == set(expected_unique)

    def test_temporal_kg_data_custom_timestamp_col(
        self, kg_config: dict, sample_temporal_facts: Tensor
    ) -> None:
        """Test that TemporalKGData works with custom timestamp column index."""
        # Create facts with timestamp in different column
        facts_reordered = torch.stack(
            [
                sample_temporal_facts[:, 0],  # source
                sample_temporal_facts[:, 3],  # timestamp (moved to col 1)
                sample_temporal_facts[:, 2],  # target
                sample_temporal_facts[:, 1],  # relation (moved to col 3)
            ],
            dim=1,
        )

        num_nodes = kg_config.get("num_nodes")
        kg_data = KGData.from_facts(facts_reordered, num_nodes=num_nodes, relabel_nodes=True)
        temporal_kg = TemporalKGData(kg_data, edge_attr_timestamp_col=0)

        expected_timestamps = sample_temporal_facts[:, 3]
        assert torch.equal(temporal_kg.edge_timestamps, expected_timestamps)

    def test_temporal_kg_data_num_entities(self, temporal_kg_data: TemporalKGData) -> None:
        """Test that num_entities property returns correct value."""
        expected_num_entities = temporal_kg_data.data.n_id.size(0)
        assert temporal_kg_data.num_entities == expected_num_entities

    def test_temporal_kg_data_list_timestamps(self, temporal_kg_data: TemporalKGData) -> None:
        """Test that list_timestamps returns list of unique timestamps."""
        timestamps = temporal_kg_data.list_timestamps()

        assert isinstance(timestamps, list)
        assert len(timestamps) == len(temporal_kg_data.unique_timestamps)
        assert set(timestamps) == set(temporal_kg_data.unique_timestamps)

    def test_temporal_kg_data_has_n_id_with_n_id(self, temporal_kg_data: TemporalKGData) -> None:
        """Test that has_n_id returns True when n_id exists."""
        assert temporal_kg_data.has_n_id() is True
        assert has_attr_without_none(temporal_kg_data.data, "n_id")

    def test_temporal_kg_data_has_n_id_without_n_id(
        self, kg_config: dict, sample_temporal_facts: Tensor
    ) -> None:
        """Test that has_n_id returns False when n_id does not exist."""
        num_nodes = kg_config.get("num_nodes")
        kg_data = KGData.from_facts(sample_temporal_facts, num_nodes=num_nodes, relabel_nodes=False)
        temporal_kg = TemporalKGData(kg_data, edge_attr_timestamp_col=1)

        assert temporal_kg.has_n_id() is False


class TestTemporalKGDataGetNodeData(TestBaseKGData):
    """Test class for get_node_data method."""

    __test__ = True

    def test_get_node_data_basic(self, temporal_kg_data: TemporalKGData) -> None:
        """Test that get_node_data returns correct node features."""
        # Get some valid node IDs
        valid_node_ids = torch.tensor([0, 1, 2], dtype=torch.long)

        node_data = temporal_kg_data.get_node_data(valid_node_ids, safe=False)

        assert node_data.shape[0] == len(valid_node_ids)
        assert node_data.shape[1] == temporal_kg_data.data.x.shape[1]
        assert torch.equal(node_data[0], temporal_kg_data.data.x[0])
        assert torch.equal(node_data[1], temporal_kg_data.data.x[1])
        assert torch.equal(node_data[2], temporal_kg_data.data.x[2])

    def test_get_node_data_safe_mode_valid(self, temporal_kg_data: TemporalKGData) -> None:
        """Test that get_node_data with safe=True works for valid node IDs."""
        num_nodes = temporal_kg_data.data.x.size(0)
        valid_node_ids = torch.tensor([0, num_nodes // 2, num_nodes - 1], dtype=torch.long)

        node_data = temporal_kg_data.get_node_data(valid_node_ids, safe=True)

        assert node_data.shape[0] == len(valid_node_ids)

    def test_get_node_data_safe_mode_invalid_negative(
        self, temporal_kg_data: TemporalKGData
    ) -> None:
        """Test that get_node_data with safe=True raises error for negative node IDs."""
        invalid_node_ids = torch.tensor([-1, 0, 1], dtype=torch.long)

        with pytest.raises(IndexError):
            temporal_kg_data.get_node_data(invalid_node_ids, safe=True)

    def test_get_node_data_safe_mode_invalid_out_of_bounds(
        self, temporal_kg_data: TemporalKGData
    ) -> None:
        """Test that get_node_data with safe=True raises error for out-of-bounds node IDs."""
        num_nodes = temporal_kg_data.data.x.size(0)
        invalid_node_ids = torch.tensor([0, num_nodes, num_nodes + 1], dtype=torch.long)

        with pytest.raises(IndexError):
            temporal_kg_data.get_node_data(invalid_node_ids, safe=True)

    def test_get_node_data_preserves_order(self, temporal_kg_data: TemporalKGData) -> None:
        """Test that get_node_data preserves the order of input node IDs."""
        node_ids = torch.tensor([5, 2, 8, 1], dtype=torch.long)

        node_data = temporal_kg_data.get_node_data(node_ids, safe=False)

        assert node_data.shape[0] == len(node_ids)
        for i, node_id in enumerate(node_ids):
            assert torch.equal(node_data[i], temporal_kg_data.data.x[node_id])


class TestTemporalKGDataGetTimestampData(TestBaseKGData):
    """Test class for get_timestamp_data method."""

    __test__ = True

    def test_get_timestamp_data_basic(self, temporal_kg_data: TemporalKGData) -> None:
        """Test that get_timestamp_data returns correct data for a timestamp."""
        timestamp_idx = temporal_kg_data.unique_timestamps[0]

        node_data, n_id, edge_index, edge_attr = temporal_kg_data.get_timestamp_data(
            timestamp_idx, relabel_nodes=True
        )

        assert node_data.shape[0] == len(n_id)
        assert edge_index.shape[0] == 2
        assert edge_index.shape[1] == edge_attr.shape[0]
        assert len(n_id) > 0

    def test_get_timestamp_data_relabel_nodes_true(self, temporal_kg_data: TemporalKGData) -> None:
        """Test that get_timestamp_data relabels nodes when relabel_nodes=True."""
        timestamp_idx = temporal_kg_data.unique_timestamps[0]

        _node_data, n_id, edge_index, _edge_attr = temporal_kg_data.get_timestamp_data(
            timestamp_idx, relabel_nodes=True
        )

        max_idx = int(edge_index.max().item())
        assert max_idx < len(n_id)
        assert edge_index.min().item() >= 0

    def test_get_timestamp_data_relabel_nodes_false(self, temporal_kg_data: TemporalKGData) -> None:
        """Test that get_timestamp_data preserves global IDs when relabel_nodes=False."""
        timestamp_idx = temporal_kg_data.unique_timestamps[0]

        _node_data, n_id, edge_index, _edge_attr = temporal_kg_data.get_timestamp_data(
            timestamp_idx, relabel_nodes=False
        )

        # edge_index should contain original global node IDs
        assert torch.all(torch.isin(edge_index.flatten(), n_id))

    def test_get_timestamp_data_correct_edges(self, temporal_kg_data: TemporalKGData) -> None:
        """Test that get_timestamp_data filters edges correctly by timestamp."""
        timestamp_idx = temporal_kg_data.unique_timestamps[0]

        # Get edges for this timestamp
        edge_mask = temporal_kg_data.edge_timestamps == timestamp_idx
        expected_edges = temporal_kg_data.data.edge_index[:, edge_mask]

        _node_data, _n_id, edge_index, _edge_attr = temporal_kg_data.get_timestamp_data(
            timestamp_idx, relabel_nodes=False
        )

        # Should have same number of edges
        assert edge_index.shape[1] == expected_edges.shape[1]

    def test_get_timestamp_data_invalid_timestamp(self, temporal_kg_data: TemporalKGData) -> None:
        """Test that get_timestamp_data raises error for invalid timestamp."""
        # Find an invalid timestamp (not in unique_timestamps)
        max_timestamp = max(temporal_kg_data.unique_timestamps)
        invalid_timestamp = max_timestamp + 100

        with pytest.raises(ValueError, match="is not valid"):
            temporal_kg_data.get_timestamp_data(invalid_timestamp)

    def test_get_timestamp_data_node_data_correctness(
        self, temporal_kg_data: TemporalKGData
    ) -> None:
        """Test that get_timestamp_data returns correct node features."""
        timestamp_idx = temporal_kg_data.unique_timestamps[0]

        node_data, n_id, _edge_index, _edge_attr = temporal_kg_data.get_timestamp_data(
            timestamp_idx, relabel_nodes=True
        )

        expected_node_data = temporal_kg_data.data.x[n_id]
        assert torch.equal(node_data, expected_node_data)

    def test_get_timestamp_data_all_timestamps(self, temporal_kg_data: TemporalKGData) -> None:
        """Test that get_timestamp_data works for all unique timestamps."""
        for timestamp_idx in temporal_kg_data.unique_timestamps:
            node_data, n_id, edge_index, edge_attr = temporal_kg_data.get_timestamp_data(
                timestamp_idx, relabel_nodes=True
            )

            assert node_data.shape[0] == len(n_id)
            assert edge_index.shape[1] == edge_attr.shape[0]
            assert len(n_id) > 0


class TestTemporalKGDataLocalToGlobal(TestBaseKGData):
    """Test class for local_to_global method."""

    __test__ = True

    def test_local_to_global_basic(self, temporal_kg_data: TemporalKGData) -> None:
        """Test that local_to_global maps local indices to global IDs correctly."""
        if not temporal_kg_data.has_n_id():
            pytest.skip("n_id not available for this test")

        local_ids = torch.tensor([0, 1, 2], dtype=torch.long)
        n_id = temporal_kg_data.data.n_id

        global_ids = temporal_kg_data.local_to_global(local_ids, n_id=n_id, safe=False)

        assert len(global_ids) == len(local_ids)
        assert torch.equal(global_ids, n_id[local_ids])

    def test_local_to_global_uses_default_n_id(self, temporal_kg_data: TemporalKGData) -> None:
        """Test that local_to_global uses self.data.n_id when n_id=None."""
        if not temporal_kg_data.has_n_id():
            pytest.skip("n_id not available for this test")

        local_ids = torch.tensor([0, 1, 2], dtype=torch.long)

        global_ids_with_none = temporal_kg_data.local_to_global(local_ids, n_id=None, safe=False)
        global_ids_explicit = temporal_kg_data.local_to_global(
            local_ids, n_id=temporal_kg_data.data.n_id, safe=False
        )

        assert torch.equal(global_ids_with_none, global_ids_explicit)

    def test_local_to_global_safe_mode_valid(self, temporal_kg_data: TemporalKGData) -> None:
        """Test that local_to_global with safe=True works for valid local IDs."""
        if not temporal_kg_data.has_n_id():
            pytest.skip("n_id not available for this test")

        n_id = temporal_kg_data.data.n_id
        max_local = len(n_id) - 1
        local_ids = torch.tensor([0, max_local // 2, max_local], dtype=torch.long)

        global_ids = temporal_kg_data.local_to_global(local_ids, n_id=n_id, safe=True)

        assert len(global_ids) == len(local_ids)

    def test_local_to_global_safe_mode_invalid_negative(
        self, temporal_kg_data: TemporalKGData
    ) -> None:
        """Test that local_to_global with safe=True raises error for negative local IDs."""
        if not temporal_kg_data.has_n_id():
            pytest.skip("n_id not available for this test")

        n_id = temporal_kg_data.data.n_id
        invalid_local_ids = torch.tensor([-1, 0, 1], dtype=torch.long)

        with pytest.raises(IndexError, match="Invalid local IDs found"):
            temporal_kg_data.local_to_global(invalid_local_ids, n_id=n_id, safe=True)

    def test_local_to_global_safe_mode_invalid_out_of_bounds(
        self, temporal_kg_data: TemporalKGData
    ) -> None:
        """Test that local_to_global with safe=True raises error for out-of-bounds local IDs."""
        if not temporal_kg_data.has_n_id():
            pytest.skip("n_id not available for this test")

        n_id = temporal_kg_data.data.n_id
        max_local = len(n_id) - 1
        invalid_local_ids = torch.tensor([0, max_local + 1, max_local + 2], dtype=torch.long)

        with pytest.raises(IndexError, match="Invalid local IDs found"):
            temporal_kg_data.local_to_global(invalid_local_ids, n_id=n_id, safe=True)

    def test_local_to_global_without_n_id(
        self, kg_config: dict, sample_temporal_facts: Tensor
    ) -> None:
        """Test that local_to_global returns local_ids unchanged when n_id is None."""
        num_nodes = kg_config.get("num_nodes")
        kg_data = KGData.from_facts(sample_temporal_facts, num_nodes=num_nodes, relabel_nodes=False)
        temporal_kg = TemporalKGData(kg_data, edge_attr_timestamp_col=1)

        local_ids = torch.tensor([0, 1, 2], dtype=torch.long)

        result = temporal_kg.local_to_global(local_ids, n_id=None, safe=False)

        assert torch.equal(result, local_ids)

    def test_local_to_global_preserves_order(self, temporal_kg_data: TemporalKGData) -> None:
        """Test that local_to_global preserves the order of input local IDs."""
        if not temporal_kg_data.has_n_id():
            pytest.skip("n_id not available for this test")

        n_id = temporal_kg_data.data.n_id
        local_ids = torch.tensor([3, 0, 5, 1], dtype=torch.long)

        global_ids = temporal_kg_data.local_to_global(local_ids, n_id=n_id, safe=False)

        assert len(global_ids) == len(local_ids)
        for i, local_id in enumerate(local_ids):
            assert global_ids[i] == n_id[local_id]


class TestTemporalKGDataGlobalToLocal(TestBaseKGData):
    """Test class for global_to_local method."""

    __test__ = True

    def test_global_to_local_basic(self, temporal_kg_data: TemporalKGData) -> None:
        """Test that global_to_local maps global IDs to local indices correctly."""
        if not temporal_kg_data.has_n_id():
            pytest.skip("n_id not available for this test")

        n_id = temporal_kg_data.data.n_id
        global_ids = n_id[:3]  # Get first 3 global IDs

        local_ids = temporal_kg_data.global_to_local(global_ids, n_id=n_id, safe=False)

        assert len(local_ids) == len(global_ids)
        # Should map to indices [0, 1, 2]
        expected_local_ids = torch.tensor([0, 1, 2], dtype=torch.long)
        assert torch.equal(local_ids, expected_local_ids)

    def test_global_to_local_uses_default_n_id(self, temporal_kg_data: TemporalKGData) -> None:
        """Test that global_to_local uses self.data.n_id when n_id=None."""
        if not temporal_kg_data.has_n_id():
            pytest.skip("n_id not available for this test")

        n_id = temporal_kg_data.data.n_id
        global_ids = n_id[:3]

        local_ids_with_none = temporal_kg_data.global_to_local(global_ids, n_id=None, safe=False)
        local_ids_explicit = temporal_kg_data.global_to_local(global_ids, n_id=n_id, safe=False)

        assert torch.equal(local_ids_with_none, local_ids_explicit)

    def test_global_to_local_safe_mode_valid(self, temporal_kg_data: TemporalKGData) -> None:
        """Test that global_to_local with safe=True works for valid global IDs."""
        if not temporal_kg_data.has_n_id():
            pytest.skip("n_id not available for this test")

        n_id = temporal_kg_data.data.n_id
        global_ids = n_id[:3]

        local_ids = temporal_kg_data.global_to_local(global_ids, n_id=n_id, safe=True)

        assert len(local_ids) == len(global_ids)

    def test_global_to_local_safe_mode_invalid(self, temporal_kg_data: TemporalKGData) -> None:
        """Test that global_to_local with safe=True raises error for invalid global IDs."""
        if not temporal_kg_data.has_n_id():
            pytest.skip("n_id not available for this test")

        n_id = temporal_kg_data.data.n_id
        # Create global IDs that don't exist in n_id
        max_global = n_id.max().item()
        invalid_global_ids = torch.tensor([max_global + 100, max_global + 200], dtype=torch.long)

        with pytest.raises(ValueError):
            temporal_kg_data.global_to_local(invalid_global_ids, n_id=n_id, safe=True)

    def test_global_to_local_without_n_id(
        self, kg_config: dict, sample_temporal_facts: Tensor
    ) -> None:
        """Test that global_to_local returns global_ids unchanged when n_id is None."""
        num_nodes = kg_config.get("num_nodes")
        kg_data = KGData.from_facts(sample_temporal_facts, num_nodes=num_nodes, relabel_nodes=False)
        temporal_kg = TemporalKGData(kg_data, edge_attr_timestamp_col=1)

        global_ids = torch.tensor([0, 1, 2], dtype=torch.long)

        result = temporal_kg.global_to_local(global_ids, n_id=None, safe=False)

        assert torch.equal(result, global_ids)

    def test_global_to_local_preserves_order(self, temporal_kg_data: TemporalKGData) -> None:
        """Test that global_to_local preserves the order of input global IDs."""
        if not temporal_kg_data.has_n_id():
            pytest.skip("n_id not available for this test")

        n_id = temporal_kg_data.data.n_id
        # Use non-sequential global IDs
        global_ids = torch.tensor([n_id[5], n_id[0], n_id[3], n_id[1]], dtype=torch.long)

        local_ids = temporal_kg_data.global_to_local(global_ids, n_id=n_id, safe=False)

        assert len(local_ids) == len(global_ids)
        # Verify mapping is correct
        for i, global_id in enumerate(global_ids):
            expected_local = torch.searchsorted(n_id, global_id, right=False)
            assert local_ids[i] == expected_local

    def test_global_to_local_round_trip(self, temporal_kg_data: TemporalKGData) -> None:
        """Test round-trip conversion: global -> local -> global."""
        if not temporal_kg_data.has_n_id():
            pytest.skip("n_id not available for this test")

        n_id = temporal_kg_data.data.n_id
        original_global_ids = n_id[:5]

        # Convert to local
        local_ids = temporal_kg_data.global_to_local(original_global_ids, n_id=n_id, safe=True)

        # Convert back to global
        restored_global_ids = temporal_kg_data.local_to_global(local_ids, n_id=n_id, safe=True)

        assert torch.equal(original_global_ids, restored_global_ids)


class TestTemporalKGDataEdgeCases(TestBaseKGData):
    """Test class for edge cases and error handling."""

    __test__ = True

    def test_temporal_kg_data_single_timestamp(self, kg_config: dict) -> None:
        """Test TemporalKGData with facts that all have the same timestamp."""
        # Create facts with single timestamp
        num_nodes = kg_config.get("num_nodes")
        num_relations = kg_config.get("num_relations")
        num_facts = 10

        subjects = torch.randint(0, num_nodes, (num_facts,))
        relations = torch.randint(0, num_relations, (num_facts,))
        objects = torch.randint(0, num_nodes, (num_facts,))
        timestamps = torch.zeros((num_facts,), dtype=torch.long)  # All same timestamp

        single_timestamp_facts = torch.stack([subjects, relations, objects, timestamps], dim=1)

        kg_data = KGData.from_facts(single_timestamp_facts, num_nodes=num_nodes, relabel_nodes=True)
        temporal_kg = TemporalKGData(kg_data, edge_attr_timestamp_col=1)

        assert len(temporal_kg.unique_timestamps) == 1
        assert temporal_kg.unique_timestamps[0] == 0

    def test_get_timestamp_data_single_edge(self, temporal_kg_data: TemporalKGData) -> None:
        """Test get_timestamp_data when timestamp has only one edge."""
        timestamp_idx = temporal_kg_data.unique_timestamps[0]

        node_data, n_id, edge_index, edge_attr = temporal_kg_data.get_timestamp_data(
            timestamp_idx, relabel_nodes=True
        )

        # Should work even with single edge
        assert edge_index.shape[1] >= 1
        assert edge_attr.shape[0] >= 1

    def test_local_to_global_empty_tensor(self, temporal_kg_data: TemporalKGData) -> None:
        """Test local_to_global with empty tensor."""
        if not temporal_kg_data.has_n_id():
            pytest.skip("n_id not available for this test")

        n_id = temporal_kg_data.data.n_id
        empty_local_ids = torch.tensor([], dtype=torch.long)

        global_ids = temporal_kg_data.local_to_global(empty_local_ids, n_id=n_id, safe=False)

        assert len(global_ids) == 0

    def test_global_to_local_empty_tensor(self, temporal_kg_data: TemporalKGData) -> None:
        """Test global_to_local with empty tensor."""
        if not temporal_kg_data.has_n_id():
            pytest.skip("n_id not available for this test")

        n_id = temporal_kg_data.data.n_id
        empty_global_ids = torch.tensor([], dtype=torch.long)

        local_ids = temporal_kg_data.global_to_local(empty_global_ids, n_id=n_id, safe=False)

        assert len(local_ids) == 0
