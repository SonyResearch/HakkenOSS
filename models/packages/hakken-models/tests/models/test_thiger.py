from typing import Any

import polars as pl
import pytest
import torch
from hakken_ml_toolkit.ml_base_structures.data_generator import DummyDataGenerator
from torch import Tensor, nn

from hakken_models.core.entities.kg_data import KGData
from hakken_models.core.entities.temporal_kg_data import TemporalKGData
from hakken_models.models.thiger import THiGER

# Test configurations
GNN_CONFIGS = [
    {"name": "GraphSAGE", "kwargs": {"hidden_channels": 64, "num_layers": 2}},
    {
        "name": "GCN",
        "kwargs": {
            "hidden_channels": 64,
            "num_layers": 2,
            "dropout": 0.0,
            "act": "relu",
        },
    },
]
DEVICE_CONFIGS = ["cuda"]

TX_CONFIGS = [
    {
        "name": "Transformer",
        "kwargs": {
            "num_heads": 8,
            "num_layers": 2,
            "dropout": 0.1,
            "use_pos_encoding": True,
            "aggregation": "cls_token",
        },
    },
    {
        "name": "Transformer",
        "kwargs": {
            "num_heads": 4,
            "num_layers": 1,
            "dropout": 0.0,
            "use_pos_encoding": False,
            "aggregation": "cls_token",
        },
    },
    {
        "name": "Transformer",
        "kwargs": {
            "num_heads": 4,
            "num_layers": 2,
            "dropout": 0.1,
            "use_pos_encoding": True,
            "aggregation": "attention",
        },
    },
]

THIGER_CONFIGS = [
    {
        "entity_embedding_dim": 64,
        "relation_embedding_dim": 64,
        "has_logits": True,
        "domain_embedding_dim": None,
    },
    {
        "entity_embedding_dim": 32,
        "relation_embedding_dim": 32,
        "has_logits": False,
        "domain_embedding_dim": None,
    },
    {
        "entity_embedding_dim": 128,
        "relation_embedding_dim": 64,
        "has_logits": True,
        "domain_embedding_dim": 32,
    },
    {
        "entity_embedding_dim": 64,
        "relation_embedding_dim": 64,
        "has_logits": True,
        "domain_embedding_dim": None,  # Should default to entity_embedding_dim
    },
]
DATA_CONFIGS = [
    {"num_entities": 100, "num_relations": 20, "num_timestamps": 1, "num_facts": 500},
    {
        "num_entities": 10_000,
        "num_relations": 100,
        "num_timestamps": 80,
        "num_facts": 100_000,
        "num_domains": 10,
    },
    {
        "num_entities": 50,
        "num_relations": 10,
        "num_timestamps": 5,
        "num_facts": 1000,
        "num_domains": 4,
    },
]


class TestTHiGER:
    __test__ = True

    @pytest.fixture(params=GNN_CONFIGS)
    def gnn_config(self, request: pytest.FixtureRequest) -> dict[str, Any]:
        """Parametrized configuration fixture for GNN models."""
        return request.param  # type: ignore

    @pytest.fixture(params=TX_CONFIGS)
    def transformer_config(self, request: pytest.FixtureRequest) -> dict[str, Any]:
        """Parametrized configuration fixture for Transformer models."""
        return request.param  # type: ignore

    @pytest.fixture(params=DEVICE_CONFIGS)
    def device(self, request: pytest.FixtureRequest) -> str:
        """Parametrized configuration fixture for data configs."""
        return request.param  # type: ignore

    @pytest.fixture(params=DATA_CONFIGS)
    def data_config(self, request: pytest.FixtureRequest) -> dict[str, Any]:
        """Parametrized configuration fixture for data configs."""
        return request.param  # type: ignore

    @pytest.fixture(params=THIGER_CONFIGS)
    def thiger_config(self, request: pytest.FixtureRequest) -> dict[str, Any]:
        """Parametrized configuration fixture for THiGER models."""
        return request.param  # type: ignore

    @pytest.fixture()
    def thiger_model(
        self,
        gnn_config: dict,
        transformer_config: dict,
        thiger_config: dict,
        data_config: dict,
        device: str,
    ) -> THiGER:
        """Parametrized fixture that instantiates THiGER model."""
        num_domains = data_config.get("num_domains")
        domain_embedding_dim = thiger_config.get("domain_embedding_dim")
        if num_domains is None:
            domain_embedding_dim = None
        return THiGER(
            num_entities=data_config["num_entities"],
            num_relations=data_config["num_relations"],
            num_timestamps=data_config["num_timestamps"],
            gnn_name=gnn_config["name"],
            gnn_kwargs=gnn_config["kwargs"],
            transformer_name=transformer_config["name"],
            transformer_kwargs=transformer_config["kwargs"],
            has_logits=thiger_config["has_logits"],
            entity_embedding_dim=thiger_config["entity_embedding_dim"],
            relation_embedding_dim=thiger_config["relation_embedding_dim"],
            domain_embedding_dim=domain_embedding_dim,
            num_domains=num_domains,
        ).to(device)

    @pytest.fixture()
    def temporal_kg(self, data_config: dict, device: str) -> KGData:
        """Fixture that creates a temporal KG for testing."""
        num_domains = data_config.get("num_domains")
        num_entities = data_config["num_entities"]

        facts = DummyDataGenerator.facts_batch(
            batch_size=10_000,
            num_entities=data_config["num_entities"],
            num_relations=data_config["num_relations"],
            num_timestamps=data_config["num_timestamps"],
            device=device,
            seed=42,
        )
        domains_mapping_df: pl.DataFrame | None = None
        if num_domains is not None:
            node_ids = list(range(num_entities))
            domains = [node_id % num_domains for node_id in node_ids]
            domains_mapping_df = pl.DataFrame({"node_id": node_ids, "domain_id": domains})

        return KGData.from_facts(
            facts=facts,
            domains_mapping_df=domains_mapping_df,
            num_nodes=num_entities,
            num_domains=num_domains,
            relabel_nodes=True,
        )

    @pytest.fixture()
    def entities(self, data_config: dict, device: str) -> Tensor:
        num_entities = data_config.get("num_entities")
        batch_size = 5
        return DummyDataGenerator.entity_batch(
            batch_size=batch_size, num_entities=num_entities, seed=42, device=device
        ).unique()  # type: ignore

    @pytest.fixture()
    def entity_pair_batch(self, data_config: dict, device: str) -> KGData:
        num_entities = data_config.get("num_entities")
        batch_size = 5
        return DummyDataGenerator.so_batch(
            batch_size=batch_size, num_entities=num_entities, seed=42, device=device
        )

    def test_model_initialization(
        self,
        data_config: dict,
        thiger_config: dict,
        thiger_model: THiGER,
    ) -> None:
        """Test that THiGER model initializes correctly."""
        assert isinstance(thiger_model, THiGER)
        assert isinstance(thiger_model, nn.Module)

        assert thiger_model.num_entities == data_config["num_entities"]
        assert thiger_model.num_relations == data_config["num_relations"]
        assert thiger_model.num_timestamps == data_config["num_timestamps"]
        assert thiger_model.entity_embedding_dim == thiger_config["entity_embedding_dim"]
        assert thiger_model.relation_embedding_dim == thiger_config["relation_embedding_dim"]
        assert thiger_model.has_logits == thiger_config["has_logits"]

    def test_embeddings_initialization(
        self,
        data_config: dict,
        thiger_config: dict,
        thiger_model: THiGER,
    ) -> None:
        """Test that all embeddings are initialized correctly."""
        assert isinstance(thiger_model.entity_embeddings, nn.Embedding)
        assert thiger_model.entity_embeddings.num_embeddings == data_config["num_entities"]
        assert thiger_model.entity_embeddings.embedding_dim == thiger_config["entity_embedding_dim"]

        assert isinstance(thiger_model.relation_embeddings, nn.Embedding)
        assert thiger_model.relation_embeddings.num_embeddings == data_config["num_relations"]
        assert (
            thiger_model.relation_embeddings.embedding_dim
            == thiger_config["relation_embedding_dim"]
        )

        if data_config.get("num_domains") is not None:
            assert thiger_model.domain_embeddings is not None
            assert isinstance(thiger_model.domain_embeddings, nn.Embedding)
            assert thiger_model.domain_embeddings.num_embeddings == data_config["num_domains"]
            expected_domain_dim = (
                thiger_config.get("domain_embedding_dim") or thiger_config["entity_embedding_dim"]
            )
            assert thiger_model.domain_embeddings.embedding_dim == expected_domain_dim
        else:
            assert thiger_model.domain_embeddings is None

    def test_node_embedding_dim_calculation(
        self,
        data_config: dict,
        thiger_config: dict,
        thiger_model: THiGER,
    ) -> None:
        """Test that node_embedding_dim is calculated correctly."""
        expected_dim = thiger_config["entity_embedding_dim"]
        if data_config.get("num_domains") is not None:
            domain_dim = (
                thiger_config.get("domain_embedding_dim") or thiger_config["entity_embedding_dim"]
            )
            expected_dim += domain_dim
        assert thiger_model.node_embedding_dim == expected_dim

    def test_get_node_embeddings(
        self, data_config: dict, thiger_model: THiGER, entities: Tensor, device: str
    ) -> None:
        """Test get_node_embeddings method."""

        bach_size = entities.size(0)

        if data_config.get("num_domains") is not None:
            domains = torch.randint(0, data_config["num_domains"], (bach_size,), device=device)
            embeddings = thiger_model.get_node_embeddings(entities, domains)
            assert embeddings.shape == (
                bach_size,
                thiger_model.node_embedding_dim,
            )
        else:
            embeddings = thiger_model.get_node_embeddings(entities)
            assert embeddings.shape == (
                bach_size,
                thiger_model.node_embedding_dim,
            )

        assert isinstance(embeddings, Tensor)
        assert not torch.isnan(embeddings).any()
        assert not torch.isinf(embeddings).any()

    def test_context_temporal_kg_property(self, thiger_model: THiGER, temporal_kg: KGData) -> None:
        """Test context_temporal_kg property getter and setter."""
        # Initially should raise ValueError
        with pytest.raises(ValueError, match="Context temporal KG has not been set"):
            _ = thiger_model.context_temporal_kg

        # Set context
        thiger_model.set_context_temporal_kg(temporal_kg)
        assert isinstance(thiger_model.context_temporal_kg, TemporalKGData)

        # Clean context
        thiger_model.clean_context_temporal_kg()
        with pytest.raises(ValueError, match="Context temporal KG has not been set"):
            _ = thiger_model.context_temporal_kg

    def test_compute_node_embeddings_at_timestamp(
        self, thiger_model: THiGER, temporal_kg: KGData, entities: Tensor
    ) -> None:
        """Test compute_node_embeddings_at_timestamp method."""
        batch_size = entities.size(0)
        timestamp_idx = 0

        thiger_model.set_context_temporal_kg(temporal_kg)

        embeddings = thiger_model.compute_node_embeddings_at_timestamp(
            timestamp_idx=timestamp_idx,
            entity_ids=entities,
        )

        assert embeddings.shape == (
            batch_size,
            thiger_model.node_embedding_dim,
        )
        assert isinstance(embeddings, Tensor)
        assert not torch.isnan(embeddings).any()
        assert not torch.isinf(embeddings).any()

    def test_compute_all_node_embeddings(
        self, data_config: dict, thiger_model: THiGER, temporal_kg: KGData, entities: Tensor
    ) -> None:
        """Test compute_node_embeddings method across all timestamps."""
        thiger_model.set_context_temporal_kg(temporal_kg)

        batch_size = entities.cpu().shape[0]

        embeddings = thiger_model.compute_node_embeddings(entity_ids=entities)

        assert embeddings.shape == (
            data_config["num_timestamps"],
            batch_size,
            thiger_model.node_embedding_dim,
        )
        assert isinstance(embeddings, Tensor)
        assert not torch.isnan(embeddings).any()
        assert not torch.isinf(embeddings).any()

        # Test with None entities (should use all entities)
        embeddings_all = thiger_model.compute_node_embeddings(entity_ids=None)
        assert embeddings_all.shape == (
            data_config["num_timestamps"],
            data_config["num_entities"],
            thiger_model.node_embedding_dim,
        )

    def test_forward_output_shape(
        self, thiger_model: THiGER, temporal_kg: KGData, device: str
    ) -> None:
        """Test that forward pass returns correct output shape."""
        thiger_model.set_context_temporal_kg(temporal_kg)

        batch_size = 10

        num_entities = thiger_model.num_entities

        entity_pair_batch = DummyDataGenerator.so_batch(
            batch_size=batch_size, num_entities=num_entities, seed=42, device=device
        )

        output = thiger_model.forward(entity_pair_batch)

        assert output.shape == (batch_size, thiger_model.pair_embedding_dim)
        assert isinstance(output, Tensor)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_gradient_flow(
        self, thiger_model: THiGER, temporal_kg: KGData, entity_pair_batch: Tensor
    ) -> None:
        """Test that gradients flow through the model."""
        thiger_model.set_context_temporal_kg(temporal_kg)

        output = thiger_model.forward(entity_pair_batch)
        loss = output.sum()
        loss.backward()

        # Check that model parameters have gradients
        has_grad = False
        for param in thiger_model.parameters():
            if param.requires_grad and param.grad is not None:
                has_grad = True
                break
        assert has_grad, "At least one parameter should have gradients"

    def test_compute_logits(
        self,
        thiger_config: dict,
        data_config: dict,
        thiger_model: THiGER,
        temporal_kg: KGData,
        entity_pair_batch: Tensor,
    ) -> None:
        """Test compute_logits method."""
        if not thiger_config["has_logits"]:
            pytest.skip("Model does not have logits MLP")

        thiger_model.set_context_temporal_kg(temporal_kg)

        batch_size = entity_pair_batch.size(0)
        logits = thiger_model.compute_logits(entity_pair_batch)

        assert logits.shape == (batch_size, data_config["num_relations"])
        assert isinstance(logits, Tensor)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

    def test_compute_logits_raises_error_when_no_logits(
        self,
        thiger_config: dict,
        thiger_model: THiGER,
        temporal_kg: KGData,
        entity_pair_batch: Tensor,
    ) -> None:
        """Test that compute_logits raises error when has_logits=False."""
        if thiger_config["has_logits"]:
            pytest.skip("Model has logits, skipping this test")

        thiger_model.set_context_temporal_kg(temporal_kg)

        with pytest.raises(ValueError, match="Model does not have logits"):
            _ = thiger_model.compute_logits(entity_pair_batch)

    def test_model_with_domains(
        self,
        data_config: dict,
        thiger_model: THiGER,
        temporal_kg: KGData,
        entities: Tensor,
        device: str,
    ) -> None:
        """Test model behavior when domains are used."""
        if data_config.get("num_domains") is None:
            pytest.skip("Model does not use domains")

        thiger_model.set_context_temporal_kg(temporal_kg)

        batch_size = entities.size(0)

        domains = torch.randint(0, data_config["num_domains"], (batch_size,), device=device)

        embeddings = thiger_model.get_node_embeddings(entities, domains)
        assert embeddings.shape == (
            batch_size,
            thiger_model.node_embedding_dim,
        )

        # Verify domain embeddings are concatenated
        assert thiger_model.node_embedding_dim == (
            thiger_model.entity_embedding_dim + thiger_model.domain_embedding_dim
        )
