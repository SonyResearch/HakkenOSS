import math
from typing import Any, cast

import pytest
import torch
from torch import nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import LRScheduler

from hakken_models.losses.ranking_relation import RankingRelationLoss
from hakken_models.models.kge import KGE, LitKGE
from hakken_models.models.kge.lightning import (
    build_default_lit_kge_val_metric_hub,
    create_lit_kge,
)
from hakken_models.scores import score_fn_registry
from hakken_models.scores.base import ScoreFn
from hakken_models.scores.conv_kb import ConvKBScore

# ============================================================================
# Test Configuration
# ============================================================================
KGE_CONFIGS = [
    {"num_entities": 100, "num_relations": 10, "embedding_dim": 32},
    {"num_entities": 50, "num_relations": 5, "embedding_dim": 16},
]

SCORE_FN_NAMES = score_fn_registry.list_all()

LOSS_CONFIGS = [
    {"name": "MarginRankingLoss", "kwargs": {"margin": 1.0}},
]

OPTIMIZER_CONFIGS = [
    {"name": "Adam", "kwargs": {"lr": 1e-3}},
]

NEGATIVES_STRATEGIES = ["hardest", "mean"]
NUM_NEGATIVES_OPTIONS = [1, 3]


# ============================================================================
# Test Class
# ============================================================================
class TestLitKGE:
    """Test LitKGE model with different configurations."""

    @pytest.fixture(params=KGE_CONFIGS)
    def kge_config(self, request: pytest.FixtureRequest) -> dict[str, int]:
        """Parametrized KGE configuration."""
        return cast(dict, request.param)

    @pytest.fixture(params=SCORE_FN_NAMES)
    def score_fn_name(self, request: pytest.FixtureRequest) -> str:
        """Parametrized score function name."""
        return cast(str, request.param)

    @pytest.fixture
    def score_fn(self, score_fn_name: str, kge_config: dict[str, int]) -> ScoreFn:
        """Score function instance from registry."""
        cls = score_fn_registry.get(score_fn_name)
        if issubclass(cls, ConvKBScore):
            return score_fn_registry.create(score_fn_name, emb_dim=kge_config["embedding_dim"])
        return score_fn_registry.create(score_fn_name)

    @pytest.fixture
    def kge(self, kge_config: dict[str, int], score_fn: ScoreFn) -> KGE:
        """KGE model instance."""
        return KGE(
            num_entities=kge_config["num_entities"],
            num_relations=kge_config["num_relations"],
            embedding_dim=kge_config["embedding_dim"],
            score_fn=score_fn,
        )

    @pytest.fixture(params=LOSS_CONFIGS)
    def loss_config(self, request: pytest.FixtureRequest) -> dict[str, Any]:
        """Parametrized loss configuration."""
        return cast(dict, request.param)

    @pytest.fixture
    def loss_fn(self, loss_config: dict[str, Any], negatives_strategy: str) -> RankingRelationLoss:
        """Margin-only KGE loss (relation term off)."""
        return RankingRelationLoss(
            entity_loss=loss_config["name"],
            entity_loss_kwargs=loss_config.get("kwargs", {}),
            neg_strategy=negatives_strategy,
            rel_loss_weight=0.0,
        )

    @pytest.fixture(params=OPTIMIZER_CONFIGS)
    def optimizer_config(self, request: pytest.FixtureRequest) -> dict[str, Any]:
        """Parametrized optimizer configuration."""
        return cast(dict, request.param)

    @pytest.fixture(params=NEGATIVES_STRATEGIES)
    def negatives_strategy(self, request: pytest.FixtureRequest) -> str:
        """Parametrized negatives strategy."""
        return cast(str, request.param)

    @pytest.fixture(params=NUM_NEGATIVES_OPTIONS)
    def num_negatives(self, request: pytest.FixtureRequest) -> int:
        """Parametrized number of negatives."""
        return cast(int, request.param)

    @pytest.fixture
    def lit_kge(
        self,
        kge: KGE,
        loss_fn: nn.Module,
        optimizer_config: dict[str, Any],
    ) -> LitKGE:
        """LitKGE model instance."""
        optimizer_cls = getattr(torch.optim, optimizer_config["name"])
        return LitKGE(
            kge=kge,
            loss_fn=loss_fn,
            optimizer_cls=optimizer_cls,
            optimizer_kwargs=optimizer_config["kwargs"],
            val_metric_hub=build_default_lit_kge_val_metric_hub(),
        )

    @pytest.fixture
    def sample_facts(self, kge_config: dict[str, int]) -> torch.Tensor:
        """Sample facts tensor for testing."""
        num_entities = kge_config["num_entities"]
        num_relations = kge_config["num_relations"]
        batch_size = 10
        facts = torch.zeros(batch_size, 3, dtype=torch.long)
        facts[:, 0] = torch.randint(0, num_entities, (batch_size,))
        facts[:, 1] = torch.randint(0, num_relations, (batch_size,))
        facts[:, 2] = torch.randint(0, num_entities, (batch_size,))
        return facts

    @pytest.fixture
    def sample_neg_facts(self, kge_config: dict[str, int], num_negatives: int) -> torch.Tensor:
        """Sample negative facts tensor [batch_size, num_negatives, 3]."""
        num_entities = kge_config["num_entities"]
        num_relations = kge_config["num_relations"]
        batch_size = 10
        facts = torch.zeros(batch_size, num_negatives, 3, dtype=torch.long)
        facts[:, :, 0] = torch.randint(0, num_entities, (batch_size, num_negatives))
        facts[:, :, 1] = torch.randint(0, num_relations, (batch_size, num_negatives))
        facts[:, :, 2] = torch.randint(0, num_entities, (batch_size, num_negatives))
        return facts

    # ========================================================================
    # Tests for forward method
    # ========================================================================

    def test_forward_output_shape(self, lit_kge: LitKGE, sample_facts: torch.Tensor) -> None:
        """Test that forward returns correct output shape."""
        scores = lit_kge.forward(sample_facts)
        assert isinstance(scores, torch.Tensor)
        assert scores.shape == (sample_facts.shape[0],)
        assert scores.dtype in (torch.float32, torch.float64)

    def test_forward_single_fact(self, lit_kge: LitKGE) -> None:
        """Test forward with single fact."""
        single_fact = torch.tensor([[0, 1, 2]], dtype=torch.long)
        scores = lit_kge.forward(single_fact)
        assert scores.shape == (1,)

    def test_forward_empty_batch(self, lit_kge: LitKGE) -> None:
        """Test forward with empty batch."""
        empty_facts = torch.empty(0, 3, dtype=torch.long)
        scores = lit_kge.forward(empty_facts)
        assert scores.shape == (0,)

    # ========================================================================
    # Tests for configure_optimizers
    # ========================================================================

    def test_configure_optimizers_returns_optimizer(self, lit_kge: LitKGE) -> None:
        """Test that configure_optimizers returns an optimizer."""
        optimizer = lit_kge.configure_optimizers()
        assert isinstance(optimizer, Optimizer)

    def test_configure_optimizers_with_scheduler(
        self,
        kge: KGE,
        loss_fn: RankingRelationLoss,
        optimizer_config: dict[str, Any],
    ) -> None:
        """Test configure_optimizers with scheduler."""
        optimizer_cls = getattr(torch.optim, optimizer_config["name"])
        lit_kge = LitKGE(
            kge=kge,
            loss_fn=loss_fn,
            optimizer_cls=optimizer_cls,
            optimizer_kwargs=optimizer_config["kwargs"],
            scheduler_cls=torch.optim.lr_scheduler.StepLR,
            scheduler_kwargs={"step_size": 10, "gamma": 0.1},
        )
        result = lit_kge.configure_optimizers()
        assert isinstance(result, dict)
        assert "optimizer" in result
        assert "lr_scheduler" in result
        assert isinstance(result["optimizer"], Optimizer)
        assert isinstance(result["lr_scheduler"]["scheduler"], LRScheduler)

    # ========================================================================
    # Tests for forward_negatives
    # ========================================================================

    def test_forward_negatives_output_shape(
        self,
        lit_kge: LitKGE,
        sample_neg_facts: torch.Tensor,
        num_negatives: int,
    ) -> None:
        """Test that forward_negatives returns correct output shape."""
        neg_scores = lit_kge.forward_negatives(sample_neg_facts)
        assert isinstance(neg_scores, torch.Tensor)
        assert neg_scores.shape == (
            sample_neg_facts.shape[0],
            num_negatives,
        )

    def test_forward_negatives_single_fact(self, lit_kge: LitKGE, num_negatives: int) -> None:
        """Test forward_negatives with single fact."""
        single_neg = torch.tensor([[[0, 1, 2]]], dtype=torch.long).expand(-1, num_negatives, -1)
        neg_scores = lit_kge.forward_negatives(single_neg)
        assert neg_scores.shape == (1, num_negatives)

    def test_forward_negatives_empty_batch(self, lit_kge: LitKGE, num_negatives: int) -> None:
        """Test forward_negatives with empty batch."""
        empty_neg = torch.empty(0, num_negatives, 3, dtype=torch.long)
        neg_scores = lit_kge.forward_negatives(empty_neg)
        assert neg_scores.shape == (0, num_negatives)

    # ========================================================================
    # Tests for loss computation (via loss_fn)
    # ========================================================================

    def test_compute_loss_single_negative(
        self,
        lit_kge: LitKGE,
        sample_facts: torch.Tensor,
        sample_neg_facts: torch.Tensor,
        num_negatives: int,
    ) -> None:
        """Test loss_fn with single negative."""
        if num_negatives != 1:
            pytest.skip("Test only for num_negatives=1")
        pos_scores = lit_kge.forward(sample_facts)
        neg_scores = lit_kge.forward_negatives(sample_neg_facts)
        loss, _ = lit_kge.loss_fn(pos_scores, neg_scores)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar

    def test_compute_loss_multiple_negatives_hardest(
        self,
        kge: KGE,
        loss_config: dict[str, Any],
        optimizer_config: dict[str, Any],
        sample_facts: torch.Tensor,
        kge_config: dict[str, int],
    ) -> None:
        """Test loss_fn with multiple negatives using hardest strategy."""
        num_neg = 3
        optimizer_cls = getattr(torch.optim, optimizer_config["name"])
        loss_fn = RankingRelationLoss(
            entity_loss=loss_config["name"],
            entity_loss_kwargs=loss_config.get("kwargs", {}),
            neg_strategy="hardest",
            rel_loss_weight=0.0,
        )
        lit_kge = LitKGE(
            kge=kge,
            loss_fn=loss_fn,
            optimizer_cls=optimizer_cls,
            optimizer_kwargs=optimizer_config["kwargs"],
        )
        batch_size = sample_facts.shape[0]
        neg_facts = torch.zeros(batch_size, num_neg, 3, dtype=torch.long)
        neg_facts[:, :, 0] = torch.randint(0, kge_config["num_entities"], (batch_size, num_neg))
        neg_facts[:, :, 1] = torch.randint(0, kge_config["num_relations"], (batch_size, num_neg))
        neg_facts[:, :, 2] = torch.randint(0, kge_config["num_entities"], (batch_size, num_neg))
        pos_scores = lit_kge.forward(sample_facts)
        neg_scores = lit_kge.forward_negatives(neg_facts)
        loss, _ = lit_kge.loss_fn(pos_scores, neg_scores)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar

    def test_compute_loss_multiple_negatives_mean(
        self,
        kge: KGE,
        loss_config: dict[str, Any],
        optimizer_config: dict[str, Any],
        sample_facts: torch.Tensor,
        kge_config: dict[str, int],
    ) -> None:
        """Test loss_fn with multiple negatives using mean strategy."""
        num_neg = 3
        optimizer_cls = getattr(torch.optim, optimizer_config["name"])
        loss_fn = RankingRelationLoss(
            entity_loss=loss_config["name"],
            entity_loss_kwargs=loss_config.get("kwargs", {}),
            neg_strategy="mean",
            rel_loss_weight=0.0,
        )
        lit_kge = LitKGE(
            kge=kge,
            loss_fn=loss_fn,
            optimizer_cls=optimizer_cls,
            optimizer_kwargs=optimizer_config["kwargs"],
        )
        batch_size = sample_facts.shape[0]
        neg_facts = torch.zeros(batch_size, num_neg, 3, dtype=torch.long)
        neg_facts[:, :, 0] = torch.randint(0, kge_config["num_entities"], (batch_size, num_neg))
        neg_facts[:, :, 1] = torch.randint(0, kge_config["num_relations"], (batch_size, num_neg))
        neg_facts[:, :, 2] = torch.randint(0, kge_config["num_entities"], (batch_size, num_neg))
        pos_scores = lit_kge.forward(sample_facts)
        neg_scores = lit_kge.forward_negatives(neg_facts)
        loss, _ = lit_kge.loss_fn(pos_scores, neg_scores)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar

    # ========================================================================
    # Tests for training_step
    # ========================================================================

    def test_training_step_output(
        self, lit_kge: LitKGE, sample_facts: torch.Tensor, sample_neg_facts: torch.Tensor
    ) -> None:
        """Test that training_step returns a loss."""
        batch = {"positives": sample_facts, "negatives": sample_neg_facts}
        loss = lit_kge.training_step(batch, 0)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar

    def test_training_step_logs_metrics(
        self, lit_kge: LitKGE, sample_facts: torch.Tensor, sample_neg_facts: torch.Tensor
    ) -> None:
        """Test that training_step logs metrics."""
        batch = {"positives": sample_facts, "negatives": sample_neg_facts}
        lit_kge.training_step(batch, 0)

    # ========================================================================
    # Tests for validation_step
    # ========================================================================

    def test_validation_step_output(
        self, lit_kge: LitKGE, sample_facts: torch.Tensor, sample_neg_facts: torch.Tensor
    ) -> None:
        """Test that validation_step returns a loss."""
        batch = {"positives": sample_facts, "negatives": sample_neg_facts}
        loss = lit_kge.validation_step(batch, 0)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar

    def test_validation_step_logs_metrics(
        self, lit_kge: LitKGE, sample_facts: torch.Tensor, sample_neg_facts: torch.Tensor
    ) -> None:
        """Test that validation_step logs metrics."""
        batch = {"positives": sample_facts, "negatives": sample_neg_facts}
        lit_kge.validation_step(batch, 0)

    def test_validation_step_without_val_metric_hub(self) -> None:
        """Validation runs when no hub is configured."""
        kge = KGE(
            num_entities=10,
            num_relations=3,
            embedding_dim=8,
            score_fn=score_fn_registry.create("DistMultScore"),
        )
        loss_fn = RankingRelationLoss(
            entity_loss="MarginRankingLoss",
            entity_loss_kwargs={"margin": 1.0},
            neg_strategy="mean",
            rel_loss_weight=0.0,
        )
        lit = LitKGE(
            kge=kge,
            loss_fn=loss_fn,
            optimizer_cls=Adam,
            optimizer_kwargs={"lr": 1e-3},
            val_metric_hub=None,
        )
        batch = {
            "positives": torch.zeros(2, 3, dtype=torch.long),
            "negatives": torch.zeros(2, 2, 3, dtype=torch.long),
        }
        loss = lit.validation_step(batch, 0)
        assert loss.dim() == 0

    def test_val_metric_hub_accumulates_across_validation_steps(
        self, lit_kge: LitKGE, sample_facts: torch.Tensor, sample_neg_facts: torch.Tensor
    ) -> None:
        lit_kge.on_validation_epoch_start()
        batch = {"positives": sample_facts, "negatives": sample_neg_facts}
        lit_kge.validation_step(batch, 0)
        lit_kge.validation_step(batch, 0)
        assert lit_kge.val_metric_hub is not None
        mr = lit_kge.val_metric_hub.compute()["mean_rank"]
        assert torch.isfinite(mr)

    def test_on_validation_epoch_end_resets_val_metric_hub(
        self, lit_kge: LitKGE, sample_facts: torch.Tensor, sample_neg_facts: torch.Tensor
    ) -> None:
        lit_kge.on_validation_epoch_start()
        batch = {"positives": sample_facts, "negatives": sample_neg_facts}
        lit_kge.validation_step(batch, 0)
        lit_kge.on_validation_epoch_end()
        assert lit_kge.val_metric_hub is not None
        assert math.isinf(lit_kge.val_metric_hub.compute()["mean_rank"].item())

    # ========================================================================
    # Tests for initialization and validation
    # ========================================================================

    def test_ranking_relation_unknown_neg_strategy_raises(
        self,
        loss_config: dict[str, Any],
    ) -> None:
        """Unknown neg_strategy is rejected inside embedded AggregatedRankingLoss."""
        bad = RankingRelationLoss(
            entity_loss=loss_config["name"],
            entity_loss_kwargs=loss_config.get("kwargs", {}),
            neg_strategy="not_a_strategy",
            rel_loss_weight=0.0,
        )
        pos = torch.tensor([1.0, 2.0])
        neg = torch.tensor([[0.5, 0.6], [0.4, 0.3]])
        with pytest.raises(NotImplementedError, match="neg_strategy"):
            bad(pos, neg)

    def test_hyperparameters_saved(self, lit_kge: LitKGE) -> None:
        """Test that hyperparameters are saved."""
        # Lightning automatically saves hyperparameters
        assert hasattr(lit_kge, "hparams")

    # ========================================================================
    # Tests for edge cases
    # ========================================================================

    def test_compute_loss_with_zero_negatives_raises(
        self,
        kge: KGE,
        loss_fn: RankingRelationLoss,
        sample_facts: torch.Tensor,
        kge_config: dict[str, int],
    ) -> None:
        """Test loss_fn behavior with edge cases."""
        lit_kge = LitKGE(
            kge=kge,
            loss_fn=loss_fn,
            optimizer_cls=Adam,
            optimizer_kwargs={"lr": 1e-3},
        )
        batch_size = sample_facts.shape[0]
        neg_facts = torch.zeros(batch_size, 1, 3, dtype=torch.long)
        neg_facts[:, :, 0] = torch.randint(0, kge_config["num_entities"], (batch_size, 1))
        neg_facts[:, :, 1] = torch.randint(0, kge_config["num_relations"], (batch_size, 1))
        neg_facts[:, :, 2] = torch.randint(0, kge_config["num_entities"], (batch_size, 1))
        pos_scores = lit_kge.forward(sample_facts)
        neg_scores = lit_kge.forward_negatives(neg_facts)
        loss, _ = lit_kge.loss_fn(pos_scores, neg_scores)
        assert isinstance(loss, torch.Tensor)


class TestLitKGERelationAuxLoss:
    """Ranking + multi-label relation BCE (composite loss + relation_labels batch)."""

    @staticmethod
    def _small_kge() -> KGE:
        return KGE(
            num_entities=24,
            num_relations=5,
            embedding_dim=16,
            score_fn=score_fn_registry.create("DistMultScore"),
        )

    def test_training_step_composite_with_relation_labels(self) -> None:
        kge = self._small_kge()
        composite = RankingRelationLoss(
            entity_loss="MarginRankingLoss",
            entity_loss_kwargs={"margin": 1.0},
            relation_loss="BCEWithLogitsLoss",
            rel_loss_weight=0.3,
            neg_strategy="hardest",
        )
        lit = LitKGE(
            kge=kge,
            loss_fn=composite,
            optimizer_cls=Adam,
            optimizer_kwargs={"lr": 1e-3},
        )
        b = 6
        num_neg = 4
        positives = torch.randint(0, 24, (b, 3))
        positives[:, 1].clamp_(0, 4)
        neg = torch.randint(0, 24, (b, num_neg, 3))
        neg[:, :, 1].clamp_(0, 4)
        rel_labels = torch.zeros(b, 5)
        rel_labels[torch.arange(b), positives[:, 1]] = 1.0
        rel_labels[:, 0] = torch.maximum(rel_labels[:, 0], torch.tensor(1.0))
        batch = {
            "positives": positives,
            "negatives": neg,
            "relation_labels": rel_labels,
        }
        loss = lit.training_step(batch, 0)
        assert loss.dim() == 0
        loss.backward()

    def test_lit_kge_rejects_non_ranking_relation_loss(self) -> None:
        kge = self._small_kge()
        with pytest.raises(TypeError, match="RankingRelationLoss"):
            LitKGE(
                kge=kge,
                loss_fn=torch.nn.MarginRankingLoss(margin=1.0),
                optimizer_cls=Adam,
                optimizer_kwargs={"lr": 1e-3},
            )

    def test_create_lit_kge_composite_factory(self) -> None:
        kge = self._small_kge()
        lit = create_lit_kge(
            kge=kge,
            loss_config={
                "name": "RankingRelationLoss",
                "kwargs": {
                    "entity_loss": "MarginRankingLoss",
                    "entity_loss_kwargs": {"margin": 1.0},
                    "rel_loss_weight": 0.2,
                    "neg_strategy": "mean",
                },
            },
            optimizer_config={"name": "Adam", "kwargs": {"lr": 1e-3}},
        )
        assert isinstance(lit.loss_fn, RankingRelationLoss)

    def test_create_lit_kge_rejects_wrong_loss_name(self) -> None:
        kge = self._small_kge()
        with pytest.raises(ValueError, match="RankingRelationLoss"):
            create_lit_kge(
                kge=kge,
                loss_config={"name": "MarginRankingLoss", "kwargs": {"margin": 1.0}},
                optimizer_config={"name": "Adam", "kwargs": {"lr": 1e-3}},
            )
        with pytest.raises(ValueError, match="RankingRelationLoss"):
            create_lit_kge(
                kge=kge,
                loss_config={"name": "AggregatedRankingLoss", "kwargs": {}},
                optimizer_config={"name": "Adam", "kwargs": {"lr": 1e-3}},
            )
