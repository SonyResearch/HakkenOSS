from typing import Any, cast

import polars as pl
import pytest
import torch
from torch import Tensor

from hakken_models.core.entities.kg_data import KGData, assert_is_kg_data, has_attr_without_none

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
        return cast(int, request.param)

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


class TestBaseKGDataStatic(TestBaseKGData):
    def test_from_facts_output_shape(self, kg_config: dict, sample_facts: Tensor) -> None:
        """Test that from_facts returns correct output shape."""
        num_nodes = kg_config.get("num_nodes")

        kg_data = KGData.from_facts(sample_facts, num_nodes=num_nodes)

        assert_is_kg_data(kg_data)

        assert kg_data.edge_index.shape == (2, sample_facts.shape[0])

        expected_edge_attr_cols = sample_facts.shape[1] - 2
        assert kg_data.edge_attr.shape == (sample_facts.shape[0], expected_edge_attr_cols)

    def test_from_facts_edge_index_correctness(self, kg_config: dict, sample_facts: Tensor) -> None:
        """Test that edge_index correctly represents graph connectivity."""
        num_nodes = kg_config.get("num_nodes")

        kg_data = KGData.from_facts(sample_facts, num_nodes=num_nodes)

        expected_edge_index = sample_facts[:, [0, 2]].t().contiguous()
        assert torch.equal(kg_data.edge_index, expected_edge_index)

    def test_from_facts_edge_attr_correctness(self, kg_config: dict, sample_facts: Tensor) -> None:
        """Test that edge_attr correctly represents edge attributes."""
        num_nodes = kg_config.get("num_nodes")

        kg_data = KGData.from_facts(sample_facts, num_nodes=num_nodes)

        cols = [i for i in range(sample_facts.shape[1]) if i not in [0, 2]]
        expected_edge_attr = sample_facts[:, cols]
        assert torch.equal(kg_data.edge_attr, expected_edge_attr)

    def test_from_facts_node_features_default(self, kg_config: dict, sample_facts: Tensor) -> None:
        """Test that from_facts creates default node features (ones) when no domains provided."""
        num_nodes = kg_config.get("num_nodes")

        kg_data = KGData.from_facts(sample_facts, num_nodes=num_nodes)

        assert kg_data.x.shape == (num_nodes, 1)
        assert torch.all(kg_data.x == 1)
        assert kg_data.x.dtype == torch.long

    def test_from_facts_preserves_dtype(self, kg_config: dict, sample_facts: Tensor) -> None:
        """Test that from_facts preserves input dtype."""
        num_nodes = kg_config.get("num_nodes")

        kg_data = KGData.from_facts(sample_facts, num_nodes=num_nodes)

        assert kg_data.edge_index.dtype == sample_facts.dtype
        assert kg_data.edge_attr.dtype == sample_facts.dtype

    def test_from_facts_preserves_device(self, kg_config: dict, sample_facts: Tensor) -> None:
        """Test that from_facts preserves input device."""
        if torch.cuda.is_available():
            num_nodes = kg_config.get("num_nodes")
            device_facts = sample_facts.cuda()

            kg_data = KGData.from_facts(device_facts, num_nodes=num_nodes)

            assert kg_data.edge_index.device == device_facts.device
            assert kg_data.edge_attr.device == device_facts.device

    def test_from_facts_relabel_nodes_false(self, kg_config: dict, sample_facts: Tensor) -> None:
        """Test that from_facts preserves original node IDs when relabel_nodes=False."""
        num_nodes = kg_config.get("num_nodes")

        kg_data = KGData.from_facts(sample_facts, num_nodes=num_nodes, relabel_nodes=False)

        assert not has_attr_without_none(kg_data, "n_id")

    def test_from_facts_relabel_nodes_true(self, kg_config: dict, sample_facts: Tensor) -> None:
        """Test that from_facts relabels nodes when relabel_nodes=True."""
        num_nodes = kg_config.get("num_nodes")

        kg_data = KGData.from_facts(sample_facts, num_nodes=num_nodes, relabel_nodes=True)

        assert kg_data.n_id is not None
        assert has_attr_without_none(kg_data, "n_id")
        node_ids = cast(Tensor, torch.unique(sample_facts[:, [0, 2]]))
        assert torch.equal(kg_data.n_id.sort()[0], node_ids.sort()[0])

        max_idx = int(kg_data.edge_index.max().item())
        assert max_idx < len(kg_data.n_id)

    def test_from_facts_num_nodes_validation(self, sample_facts: Tensor) -> None:
        """Test that from_facts validates num_nodes is large enough."""
        max_node_id = int(sample_facts[:, [0, 2]].max().item())
        invalid_num_nodes = max_node_id

        with pytest.raises(IndexError, match="< edge_index.max\\(\\) \\+ 1"):
            KGData.from_facts(sample_facts, num_nodes=invalid_num_nodes)

    # ========================================================================
    # Domain-related tests
    # ========================================================================

    def test_from_facts_with_domains_mapping(
        self, kg_config: dict, sample_facts: Tensor, sample_domains_mapping_df: pl.DataFrame
    ) -> None:
        """Test that from_facts correctly processes domains_mapping_df."""
        num_nodes = kg_config.get("num_nodes")
        num_domains = kg_config.get("num_domains")

        kg_data = KGData.from_facts(
            sample_facts,
            domains_mapping_df=sample_domains_mapping_df,
            num_nodes=num_nodes,
        )

        assert kg_data.x is not None
        assert has_attr_without_none(kg_data, "x")
        assert kg_data.x.shape[0] <= num_nodes

        assert kg_data.x.max().item() < num_domains

    def test_from_facts_with_explicit_num_domains(
        self, kg_config: dict, sample_facts: Tensor, sample_domains_mapping_df: pl.DataFrame
    ) -> None:
        """Test that from_facts accepts explicit num_domains."""
        num_nodes = kg_config.get("num_nodes")
        num_domains = kg_config.get("num_domains")

        kg_data = KGData.from_facts(
            sample_facts,
            domains_mapping_df=sample_domains_mapping_df,
            num_nodes=num_nodes,
            num_domains=num_domains,
        )

        assert_is_kg_data(kg_data)
        assert kg_data.x.max().item() < num_domains

    def test_from_facts_rejects_num_domains_without_mapping(self, sample_facts: Tensor) -> None:
        """Test that from_facts rejects num_domains when domains_mapping_df is None."""
        with pytest.raises(ValueError, match="num_domains provided without domains_mapping_df"):
            KGData.from_facts(sample_facts, num_nodes=10, num_domains=5)

    def test_from_facts_without_domains_mapping(
        self, kg_config: dict, sample_facts: Tensor
    ) -> None:
        """Test that from_facts works correctly without domains_mapping_df."""
        num_nodes = kg_config.get("num_nodes")

        kg_data = KGData.from_facts(sample_facts, num_nodes=num_nodes)

        assert kg_data.x.shape == (num_nodes, 1)
        assert torch.all(kg_data.x == 1)

    def test_extract_node_domains(
        self, sample_facts: Tensor, sample_domains_mapping_df: pl.DataFrame
    ) -> None:
        """Test the extract_node_domains method."""
        edge_pairs = sample_facts[:, [0, 2]]
        node_ids: Tensor = edge_pairs.unique()

        node_domains = KGData.extract_node_domains(
            domains_mapping_df=sample_domains_mapping_df,
            node_ids=node_ids,
        )

        assert isinstance(node_domains, Tensor)

        node_ids_list = node_ids.tolist()
        expected_length = len(
            sample_domains_mapping_df.filter(pl.col("node_id").is_in(node_ids_list))
        )
        assert len(node_domains) == expected_length

    def test_extract_node_domains_filters_correctly(self) -> None:
        """Test that extract_node_domains only returns domains for nodes in node_ids."""
        domains_df = pl.DataFrame(
            {
                "node_id": [0, 1, 2, 3, 4, 5],
                "domain_id": [0, 1, 0, 1, 0, 1],
            }
        )

        # Query only for nodes [1, 3, 5]
        node_ids = torch.tensor([1, 3, 5])

        node_domains = KGData.extract_node_domains(
            domains_mapping_df=domains_df,
            node_ids=node_ids,
        )

        # Should only return domains for nodes 1, 3, 5
        assert isinstance(node_domains, Tensor)
        assert len(node_domains) == 3
        # Verify the domains match (order may vary, so check values)
        result_domains = set(node_domains.flatten().tolist())
        expected_domains = {1}  # All should be domain 1
        assert result_domains == expected_domains


class TestKGDataEdgeCases(TestBaseKGData):
    """Test class for edge cases and error handling."""

    __test__ = True

    def test_from_facts_empty_facts(self) -> None:
        """Test from_facts with empty facts tensor."""
        empty_facts = torch.empty((0, 3), dtype=torch.long)
        with pytest.raises(ValueError, match="facts is empty"):
            KGData.from_facts(empty_facts, num_nodes=10)

    def test_from_facts_single_edge(self) -> None:
        """Test from_facts with single edge."""
        single_fact = torch.tensor([[0, 1, 2]], dtype=torch.long)

        kg_data = KGData.from_facts(single_fact, num_nodes=3, relabel_nodes=True)

        assert kg_data.edge_index.shape == (2, 1)
        assert kg_data.edge_attr.shape == (1, 1)
        print(kg_data.edge_index)
        assert torch.equal(kg_data.edge_index, torch.tensor([[0], [1]]))
        assert torch.equal(kg_data.edge_attr, torch.tensor([[1]]))

    def test_from_facts_multiple_edge_attributes(self) -> None:
        """Test from_facts with multiple edge attributes (more than just relation)."""
        facts = torch.tensor(
            [
                [0, 1, 2, 10, 20],
                [1, 2, 3, 11, 21],
            ],
            dtype=torch.long,
        )

        kg_data = KGData.from_facts(facts, num_nodes=4)

        # edge_attr should contain columns 1, 3, 4 (relation, attr1, attr2)
        assert kg_data.edge_attr.shape == (2, 3)
        expected_edge_attr = torch.tensor(
            [
                [1, 10, 20],
                [2, 11, 21],
            ],
            dtype=torch.long,
        )
        assert torch.equal(kg_data.edge_attr, expected_edge_attr)

    def test_from_facts_sparse_node_ids(self) -> None:
        """Test from_facts with sparse node IDs."""
        # Facts with non-contiguous node IDs
        sparse_facts = torch.tensor(
            [
                [0, 0, 5],
                [5, 1, 100],
                [100, 0, 0],
            ],
            dtype=torch.long,
        )

        kg_data = KGData.from_facts(sparse_facts, num_nodes=101, relabel_nodes=False)

        assert kg_data.num_nodes == 101
        assert kg_data.edge_index.shape == (2, 3)
        unique_nodes: Tensor = torch.unique(kg_data.edge_index)
        assert set(unique_nodes.tolist()) == {0, 5, 100}
