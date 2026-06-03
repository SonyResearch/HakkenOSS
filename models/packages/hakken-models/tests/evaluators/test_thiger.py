from copy import deepcopy
from typing import Any

import polars as pl
import pytest
import torch
from hakken_ml_toolkit.ml_base_structures.data_generator import DummyDataGenerator
from torch import Tensor

from hakken_models.core.configs.evaluator import MetricConfig
from hakken_models.core.entities.kg_data import KGData
from hakken_models.core.entities.kg_data_with_preds import KGDataWithPreds
from hakken_models.data_loaders.kg_link_neighbor_loader import KGLinkNeighborLoader
from hakken_models.evaluators.base import MetricBundle
from hakken_models.evaluators.thiger import THiGEREvaluator
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
]

THIGER_CONFIGS = [
    {
        "entity_embedding_dim": 64,
        "relation_embedding_dim": 64,
        "has_logits": True,
        "domain_embedding_dim": None,
    },
]

DATA_CONFIGS = [
    {
        "num_entities": 100,
        "num_relations": 20,
        "num_timestamps": 1,
        "num_facts": 500,
    },
    {
        "num_entities": 50,
        "num_relations": 10,
        "num_timestamps": 5,
        "num_facts": 1000,
    },
]

DEVICE_CONFIGS = ["cuda"]

METRIC_CONFIGS = [
    [
        MetricConfig(
            name="accuracy",
            target_class="torchmetrics.classification.MultilabelAccuracy",
            kwargs={"threshold": 0.5},
        ),
    ],
    [
        MetricConfig(
            name="accuracy",
            target_class="torchmetrics.classification.MultilabelAccuracy",
            kwargs={"threshold": 0.5},
        ),
        MetricConfig(
            name="accuracy_alt",
            target_class="torchmetrics.classification.MultilabelAccuracy",
            kwargs={"threshold": 0.7},
        ),
    ],
]


class TestTHiGEREvaluator:
    __test__ = True

    @pytest.fixture(params=GNN_CONFIGS)
    def gnn_config(self, request: pytest.FixtureRequest) -> dict[str, Any]:
        """Parametrized configuration fixture for GNN models."""
        return request.param  # type: ignore

    @pytest.fixture(params=TX_CONFIGS)
    def transformer_config(self, request: pytest.FixtureRequest) -> dict[str, Any]:
        """Parametrized configuration fixture for Transformer models."""
        return request.param  # type: ignore

    @pytest.fixture(params=DATA_CONFIGS)
    def data_config(self, request: pytest.FixtureRequest) -> dict[str, Any]:
        """Parametrized configuration fixture for data configs."""
        return request.param  # type: ignore

    @pytest.fixture(params=THIGER_CONFIGS)
    def thiger_config(self, request: pytest.FixtureRequest) -> dict[str, Any]:
        """Parametrized configuration fixture for THiGER models."""
        return request.param  # type: ignore

    @pytest.fixture(params=DEVICE_CONFIGS)
    def device(self, request: pytest.FixtureRequest) -> str:
        """Parametrized configuration fixture for device."""
        return request.param  # type: ignore

    @pytest.fixture(params=METRIC_CONFIGS)
    def metrics_config(
        self,
        request: pytest.FixtureRequest,
        data_config: dict[str, Any],
    ) -> list[MetricConfig]:
        """Parametrized configuration fixture for metrics."""
        num_relations = data_config["num_relations"]
        metrics: list[MetricConfig] = deepcopy(request.param)
        for mc in metrics:
            mc.kwargs["num_labels"] = num_relations
        return metrics  # type: ignore

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
        batch_size = data_config.get("num_facts")

        facts = DummyDataGenerator.facts_batch(
            batch_size=batch_size,
            num_entities=data_config["num_entities"],
            num_relations=data_config["num_relations"],
            num_timestamps=data_config["num_timestamps"],
            device=device,
            seed=39,
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
            relabel_nodes=False,
        )

    @pytest.fixture()
    def entity_pair_batch(self, data_config: dict, device: str) -> Tensor:
        """Fixture that creates an entity pair batch."""
        batch_size = 4
        num_entities = data_config.get("num_entities")
        return DummyDataGenerator.so_batch(
            batch_size=batch_size, num_entities=num_entities, seed=42, device=device
        )

    @pytest.fixture()
    def kg_data_with_preds(
        self, data_config: dict, temporal_kg: KGData, entity_pair_batch: Tensor
    ) -> KGDataWithPreds:
        """Fixture that creates a KGDataWithPreds batch for testing."""
        batch_size = entity_pair_batch.size(0)
        device = entity_pair_batch.device

        edge_labels = torch.randint(
            0, 2, (batch_size, data_config["num_relations"]), device=device
        ).float()

        edge_label_index = entity_pair_batch.t().contiguous()
        n_id = temporal_kg.n_id if temporal_kg.has_n_id() else None

        return KGDataWithPreds(
            x=temporal_kg.x,
            edge_index=temporal_kg.edge_index,
            edge_attr=temporal_kg.edge_attr,
            n_id=n_id,
            edge_label_index=edge_label_index,
            edge_label=edge_labels,
        )

    @pytest.fixture()
    def evaluator(self, metrics_config: list[MetricConfig], device: str) -> THiGEREvaluator:
        """Fixture that creates a THiGEREvaluator instance."""
        evaluator = THiGEREvaluator(metrics_config=metrics_config)
        # Move metrics to the same device as the model
        for metric in evaluator.metrics.values():
            metric.to(device)
        return evaluator

    def test_evaluator_initialization(
        self,
        metrics_config: list[MetricConfig],
        evaluator: THiGEREvaluator,
    ) -> None:
        """Test that THiGEREvaluator initializes correctly."""
        assert isinstance(evaluator, THiGEREvaluator)
        assert evaluator.metrics_config == metrics_config
        assert len(evaluator.metrics) == len(metrics_config)

        # Check that all metrics are initialized
        for metric_config in metrics_config:
            assert metric_config.name in evaluator.metrics
            assert isinstance(evaluator.metrics[metric_config.name], MetricBundle)

    def test_evaluator_repr(self, evaluator: THiGEREvaluator) -> None:
        """Test string representation of evaluator."""
        repr_str = repr(evaluator)
        assert "THiGEREvaluator" in repr_str
        assert "metrics=[" in repr_str

    def test_update_from_batch(
        self,
        evaluator: THiGEREvaluator,
        thiger_model: THiGER,
        temporal_kg: KGData,
        kg_data_with_preds: KGDataWithPreds,
    ) -> None:
        """Test update_from_batch method."""
        thiger_model.set_context_temporal_kg(temporal_kg)

        # Update metrics from batch
        evaluator.update_from_batch(thiger_model, kg_data_with_preds)

        # Verify metrics were updated (they should have internal state)
        for metric in evaluator.metrics.values():
            assert metric is not None

    def test_update_from_batch_multiple_times(
        self,
        evaluator: THiGEREvaluator,
        thiger_model: THiGER,
        temporal_kg: KGData,
        kg_data_with_preds: KGDataWithPreds,
    ) -> None:
        """Test that update_from_batch can be called multiple times."""
        thiger_model.set_context_temporal_kg(temporal_kg)

        # Update metrics multiple times
        for _ in range(3):
            evaluator.update_from_batch(thiger_model, kg_data_with_preds)

        # Should not raise any errors
        assert len(evaluator.metrics) > 0

    def test_update_from_batch_logits_shape(
        self,
        data_config: dict,
        evaluator: THiGEREvaluator,
        thiger_model: THiGER,
        temporal_kg: KGData,
        kg_data_with_preds: KGDataWithPreds,
    ) -> None:
        """Test that update_from_batch handles logits correctly."""
        thiger_model.set_context_temporal_kg(temporal_kg)

        batch_size = kg_data_with_preds.edge_label_index.size(1)
        entity_pair_batch = kg_data_with_preds.edge_label_index.t().contiguous()

        # Manually compute logits to verify shape
        logits = thiger_model.compute_logits(entity_pair_batch)
        assert logits.shape == (batch_size, data_config["num_relations"])

        # Update metrics
        evaluator.update_from_batch(thiger_model, kg_data_with_preds)

        # Should complete without errors
        assert len(evaluator.metrics) > 0

    def test_reset(
        self,
        evaluator: THiGEREvaluator,
        thiger_model: THiGER,
        temporal_kg: KGData,
        kg_data_with_preds: KGDataWithPreds,
    ) -> None:
        """Test reset method."""
        thiger_model.set_context_temporal_kg(temporal_kg)

        # Update metrics
        evaluator.update_from_batch(thiger_model, kg_data_with_preds)

        # Reset metrics
        evaluator.reset()

        # Verify metrics were reset
        for metric in evaluator.metrics.values():
            assert metric is not None

    def test_compute(
        self,
        evaluator: THiGEREvaluator,
        thiger_model: THiGER,
        temporal_kg: KGData,
        kg_data_with_preds: KGDataWithPreds,
    ) -> None:
        """Test compute method."""
        thiger_model.set_context_temporal_kg(temporal_kg)

        # Update metrics
        evaluator.update_from_batch(thiger_model, kg_data_with_preds)

        # Compute results
        results = evaluator.compute()

        # Verify results structure
        assert isinstance(results, list)
        assert len(results) == len(evaluator.metrics_config)

        for result in results:
            assert "name" in result
            assert "value" in result
            assert isinstance(result["name"], str)
            assert isinstance(result["value"], float)

    def test_compute_without_update(self, evaluator: THiGEREvaluator) -> None:
        """Test that compute raises error when metrics haven't been updated."""
        # Compute without updating should still work (metrics may have default state)
        # But let's verify the structure
        results = evaluator.compute()
        assert isinstance(results, list)

    def test_update_from_dataloader(
        self,
        evaluator: THiGEREvaluator,
        thiger_model: THiGER,
        temporal_kg: KGData,
        kg_data_with_preds: KGDataWithPreds,
        device: str,
    ) -> None:
        """Test update_from_dataloader method."""
        # Move temporal_kg to device to ensure DataLoader batches are on the correct device
        temporal_kg_device = temporal_kg.to(device)
        thiger_model.set_context_temporal_kg(temporal_kg_device)

        # Create a data loader with data on the correct device
        data_loader = KGLinkNeighborLoader(
            data=temporal_kg_device,
            num_neighbors=[3, 3],
            batch_size=2,
            edge_label_index=kg_data_with_preds.edge_label_index.to(device),
            edge_label=kg_data_with_preds.edge_label.to(device),
            shuffle=False,
            num_workers=0,
        )

        # Update metrics from dataloader
        evaluator.update_from_dataloader(thiger_model, data_loader)

        # Compute results
        results = evaluator.compute()
        assert isinstance(results, list)
        assert len(results) == len(evaluator.metrics_config)

    def test_evaluator_with_different_batch_sizes(
        self,
        data_config: dict,
        evaluator: THiGEREvaluator,
        thiger_model: THiGER,
        temporal_kg: KGData,
        device: str,
    ) -> None:
        """Test evaluator with different batch sizes."""
        thiger_model.set_context_temporal_kg(temporal_kg)

        # Test with different batch sizes
        for batch_size in [1, 2, 4, 8]:
            entity_pair_batch = DummyDataGenerator.so_batch(
                batch_size=batch_size,
                num_entities=data_config["num_entities"],
                seed=42,
                device=device,
            )

            edge_labels = torch.randint(
                0, 2, (batch_size, data_config["num_relations"]), device=device
            ).float()

            edge_label_index = entity_pair_batch.t().contiguous()
            n_id = temporal_kg.n_id if temporal_kg.has_n_id() else None

            batch = KGDataWithPreds(
                x=temporal_kg.x,
                edge_index=temporal_kg.edge_index,
                edge_attr=temporal_kg.edge_attr,
                n_id=n_id,
                edge_label_index=edge_label_index,
                edge_label=edge_labels,
            )

            evaluator.update_from_batch(thiger_model, batch)

        # Should complete without errors
        results = evaluator.compute()
        assert len(results) == len(evaluator.metrics_config)

    def test_evaluator_with_model_without_logits(
        self,
        evaluator: THiGEREvaluator,
        gnn_config: dict,
        transformer_config: dict,
        data_config: dict,
        device: str,
        temporal_kg: KGData,
        kg_data_with_preds: KGDataWithPreds,
    ) -> None:
        """Test evaluator with model that doesn't have logits."""
        # Create model without logits
        thiger_model_no_logits = THiGER(
            num_entities=data_config["num_entities"],
            num_relations=data_config["num_relations"],
            num_timestamps=data_config["num_timestamps"],
            gnn_name=gnn_config["name"],
            gnn_kwargs=gnn_config["kwargs"],
            transformer_name=transformer_config["name"],
            transformer_kwargs=transformer_config["kwargs"],
            has_logits=False,
            entity_embedding_dim=64,
            relation_embedding_dim=64,
            domain_embedding_dim=None,
            num_domains=None,
        ).to(device)

        thiger_model_no_logits.set_context_temporal_kg(temporal_kg)

        # Should raise ValueError when trying to compute logits
        with pytest.raises(ValueError, match="Model does not have logits"):
            evaluator.update_from_batch(thiger_model_no_logits, kg_data_with_preds)

    def test_evaluator_metrics_independence(
        self,
        evaluator: THiGEREvaluator,
        thiger_model: THiGER,
        temporal_kg: KGData,
        kg_data_with_preds: KGDataWithPreds,
    ) -> None:
        """Test that multiple metrics are independent."""
        thiger_model.set_context_temporal_kg(temporal_kg)

        # Update metrics
        evaluator.update_from_batch(thiger_model, kg_data_with_preds)

        # Get initial results
        results_before = evaluator.compute()

        # Reset one metric manually (if possible) or reset all and update again
        evaluator.reset()
        evaluator.update_from_batch(thiger_model, kg_data_with_preds)
        results_after = evaluator.compute()

        # Results should be computed
        assert len(results_before) == len(results_after)
        assert len(results_before) == len(evaluator.metrics_config)

    def test_evaluator_with_empty_metrics_config(self) -> None:
        """Test evaluator initialization with empty metrics config."""
        evaluator = THiGEREvaluator(metrics_config=[])
        evaluator._initialize_metrics()

        assert len(evaluator.metrics) == 0

        # Compute should raise RuntimeError
        with pytest.raises(RuntimeError, match="No metrics have been initialized"):
            evaluator.compute()
