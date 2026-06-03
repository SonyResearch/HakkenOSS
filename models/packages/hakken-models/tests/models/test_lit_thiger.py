from typing import Any

import polars as pl
import pytest
import torch
from hakken_ml_toolkit.ml_base_structures.data_generator import (
    DummyDataGenerator,
)
from lightning.pytorch import Trainer
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from hakken_models.core.entities.kg_data import KGData
from hakken_models.core.entities.kg_data_with_preds import KGDataWithPreds
from hakken_models.data_loaders.kg_link_neighbor_loader import KGLinkNeighborLoader
from hakken_models.losses import loss_fn_registry
from hakken_models.models.thiger import LitTHiGER, THiGER

# Test configurations
GNN_CONFIGS = [
    {
        "name": "GraphSAGE",
        "kwargs": {"hidden_channels": 64, "num_layers": 2},
    },
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
        "num_relations": 5,
        "num_timestamps": 5,
        "num_facts": 1000,
    },
]

DEVICE_CONFIGS = ["cuda"]

LOSS_CONFIGS = [
    {"name": "BCEWithLogitsLoss", "kwargs": {}},
    {"name": "BCEWithLogitsLoss", "kwargs": {"reduction": "mean"}},
    {"name": "FLWithLogitsLoss", "kwargs": {"gamma": 2.0}},
    {
        "name": "FLWithLogitsLoss",
        "kwargs": {"gamma": 1.0, "reduction": "mean"},
    },
    {
        "name": "FLWithLogitsLoss",
        "kwargs": {
            "gamma": 2.0,
            "regularization_coeff": 0.1,
            "reduction": "mean",
        },
    },
]

OPTIM_CONFIGS = [
    {"name": "Adam", "kwargs": {"lr": 0.001}},
]


SCHEDULER_CONFIGS = [
    {"name": "StepLR", "kwargs": {"step_size": 10, "gamma": 0.1}},
    {
        "name": "CosineAnnealingLR",
        "kwargs": {
            "T_max": 10,
            "eta_min": 1e-5,
        },
    },
    None,
]


class TestLitTHiGER:
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

    @pytest.fixture(params=LOSS_CONFIGS)
    def loss_config(self, request: pytest.FixtureRequest) -> dict[str, Any]:
        """Parametrized configuration fixture for loss functions."""
        return request.param  # type: ignore

    @pytest.fixture(params=OPTIM_CONFIGS)
    def optimizer_config(self, request: pytest.FixtureRequest) -> dict[str, Any]:
        """Parametrized configuration fixture for optimizers."""
        return request.param  # type: ignore

    @pytest.fixture(params=SCHEDULER_CONFIGS)
    def scheduler_config(self, request: pytest.FixtureRequest) -> dict[str, Any] | None:
        """Parametrized configuration fixture for schedulers."""
        return request.param  # type: ignore

    @pytest.fixture(params=DEVICE_CONFIGS)
    def device(self, request: pytest.FixtureRequest) -> str:
        """Parametrized configuration fixture for device."""
        return request.param  # type: ignore

    @pytest.fixture()
    def thiger_model(
        self,
        gnn_config: dict,
        transformer_config: dict,
        thiger_config: dict,
        data_config: dict,
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
        )

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
    def loss_fn(self, loss_config: dict) -> nn.Module:
        """Fixture that creates a loss function from config."""
        loss_fn_cls = loss_fn_registry.get(loss_config["name"])
        return loss_fn_cls(**loss_config.get("kwargs", {}))

    @pytest.fixture()
    def lit_thiger(
        self,
        thiger_model: THiGER,
        loss_fn: nn.Module,
        optimizer_config: dict,
        scheduler_config: dict[str, Any] | None,
        device: str,
    ) -> LitTHiGER:
        """Fixture that creates a LitTHiGER instance."""
        optimizer_cls = getattr(torch.optim, optimizer_config["name"])

        scheduler_cls = None
        scheduler_kwargs = None
        if scheduler_config:
            scheduler_cls = getattr(torch.optim.lr_scheduler, scheduler_config["name"])
            scheduler_kwargs = scheduler_config.get("kwargs", {})

        return LitTHiGER(
            thiger=thiger_model,
            loss_fn=loss_fn,
            optimizer_cls=optimizer_cls,
            optimizer_kwargs=optimizer_config["kwargs"],
            scheduler_cls=scheduler_cls,
            scheduler_kwargs=scheduler_kwargs,
        ).to(device)

    def test_model_initialization(
        self,
        thiger_model: THiGER,
        loss_fn: nn.Module,
        lit_thiger: LitTHiGER,
    ) -> None:
        """Test that LitTHiGER model initializes correctly."""
        assert isinstance(lit_thiger, LitTHiGER)
        assert lit_thiger.thiger is thiger_model
        assert lit_thiger.loss_fn is loss_fn
        assert lit_thiger.loss_fn_name == loss_fn.__class__.__name__
        assert isinstance(lit_thiger.thiger, THiGER)
        assert isinstance(lit_thiger.loss_fn, nn.Module)

    def test_forward_output_shape(
        self,
        data_config: dict,
        lit_thiger: LitTHiGER,
        temporal_kg: KGData,
        entity_pair_batch: Tensor,
    ) -> None:
        """Test that forward pass returns correct output shape."""
        lit_thiger.thiger.set_context_temporal_kg(temporal_kg)

        batch_size = entity_pair_batch.size(0)
        output = lit_thiger.forward(entity_pair_batch)

        assert output.shape == (batch_size, data_config["num_relations"])
        assert isinstance(output, Tensor)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_compute_loss(
        self,
        data_config: dict,
        lit_thiger: LitTHiGER,
    ) -> None:
        """Test compute_loss method."""
        batch_size = 10
        num_relations = data_config["num_relations"]

        logits = torch.randn(batch_size, num_relations)
        targets = torch.randint(0, 2, (batch_size, num_relations)).float()

        loss = lit_thiger.compute_loss(logits, targets)

        assert isinstance(loss, Tensor)
        assert loss.dim() == 0
        assert not torch.isnan(loss).any()
        assert not torch.isinf(loss).any()
        assert loss.item() >= 0

    def test_training_step(
        self,
        lit_thiger: LitTHiGER,
        temporal_kg: KGData,
        kg_data_with_preds: KGDataWithPreds,
    ) -> None:
        """Test training step for a single batch."""
        lit_thiger.thiger.set_context_temporal_kg(temporal_kg)

        loss = lit_thiger.training_step(kg_data_with_preds, 0)

        assert isinstance(loss, Tensor)
        assert loss.dim() == 0  # Scalar loss
        assert not torch.isnan(loss).any()
        assert not torch.isinf(loss).any()
        assert loss.item() >= 0  # Loss should be non-negative

    def test_validation_step(
        self,
        lit_thiger: LitTHiGER,
        temporal_kg: KGData,
        kg_data_with_preds: KGDataWithPreds,
    ) -> None:
        """Test validation step."""
        lit_thiger.thiger.set_context_temporal_kg(temporal_kg)

        loss = lit_thiger.validation_step(kg_data_with_preds, 0)

        assert isinstance(loss, Tensor)
        assert loss.dim() == 0  # Scalar loss
        assert not torch.isnan(loss).any()
        assert not torch.isinf(loss).any()
        assert loss.item() >= 0  # Loss should be non-negative

    def test_configure_optimizers_without_scheduler(
        self,
        lit_thiger: LitTHiGER,
        optimizer_config: dict,
        scheduler_config: dict[str, Any] | None,
    ) -> None:
        """Test optimizer configuration without scheduler."""
        if scheduler_config is not None:
            pytest.skip("Testing without scheduler")

        optimizer = lit_thiger.configure_optimizers()

        assert isinstance(optimizer, Optimizer)
        optimizer_cls = getattr(torch.optim, optimizer_config["name"])
        assert isinstance(optimizer, optimizer_cls)

    def test_configure_optimizers_with_scheduler(
        self,
        lit_thiger: LitTHiGER,
        scheduler_config: dict[str, Any] | None,
    ) -> None:
        """Test optimizer configuration with scheduler."""
        if scheduler_config is None:
            pytest.skip("Testing with scheduler")

        config = lit_thiger.configure_optimizers()

        assert isinstance(config, dict)
        assert "optimizer" in config
        assert "lr_scheduler" in config
        assert isinstance(config["optimizer"], Optimizer)
        assert isinstance(config["lr_scheduler"], dict)
        assert "scheduler" in config["lr_scheduler"]
        scheduler = config["lr_scheduler"]["scheduler"]
        assert isinstance(scheduler, LRScheduler)

    def test_gradient_flow(
        self,
        lit_thiger: LitTHiGER,
        temporal_kg: KGData,
        kg_data_with_preds: KGDataWithPreds,
    ) -> None:
        """Test that gradients flow through the model."""
        lit_thiger.thiger.set_context_temporal_kg(temporal_kg)

        loss = lit_thiger.training_step(kg_data_with_preds, 0)
        loss.backward()

        # Check that model parameters have gradients
        has_grad = False
        for param in lit_thiger.thiger.parameters():
            if param.requires_grad and param.grad is not None:
                has_grad = True
                break
        assert has_grad, "At least one parameter should have gradients"

    def test_trainer(
        self,
        lit_thiger: LitTHiGER,
        temporal_kg: KGData,
        kg_data_with_preds: KGDataWithPreds,
        device: str,
    ) -> None:
        """Test PyTorch Lightning Trainer with LitTHiGER."""
        # Set context temporal KG

        # Create KGLinkNeighborLoader instances
        num_neighbors = [3, 3]  # Example neighbor sampling configuration

        train_loader = KGLinkNeighborLoader(
            data=temporal_kg,
            num_neighbors=num_neighbors,
            batch_size=10,
            edge_label_index=kg_data_with_preds.edge_label_index,
            edge_label=kg_data_with_preds.edge_label,
            shuffle=True,
            num_workers=8,
        )

        val_loader = KGLinkNeighborLoader(
            data=temporal_kg,
            num_neighbors=num_neighbors,
            batch_size=10,
            edge_label_index=kg_data_with_preds.edge_label_index,
            edge_label=kg_data_with_preds.edge_label,
            shuffle=False,
            num_workers=8,
        )

        # Create trainer with minimal configuration for testing
        trainer = Trainer(
            max_epochs=10,
            limit_train_batches=3,
            limit_val_batches=3,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            devices=1,
            accelerator="auto" if device != "cpu" else "cpu",
            check_val_every_n_epoch=1,
            enable_checkpointing=False,
        )

        # Run training
        trainer.fit(lit_thiger, train_dataloaders=train_loader, val_dataloaders=val_loader)

        # Verify trainer completed successfully
        assert trainer.state.finished, "Trainer should have finished"
        assert trainer.current_epoch == 10, "Should have completed 1 epoch"
