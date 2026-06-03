from __future__ import annotations

import math
from typing import Any, ClassVar

import pytest
import torch
from torch import Tensor, nn

from hakken_models.core.configs.model import GNNConfig
from hakken_models.core.embedder import EmbedderConfig
from hakken_models.core.entities.kg_data import KGData, assert_is_kg_data
from hakken_models.data_loaders.temporal_kg_link_neighbor_loader import (
    corrupt_entity_pairs,
)
from hakken_models.losses import compute_pos_weight_from_relation_labels
from hakken_models.models.kge.lightning import build_default_lit_kge_val_metric_hub
from hakken_models.models.segal import (
    LitSeGAL,
    RankingRelationLoss,
    ScoringConfig,
    SeGAL,
    SeGALConfig,
    SeGALDataModule,
    TemporalEncoder,
    TemporalEncoderConfig,
    create_lit_segal,
)
from hakken_models.models.segal.base import _build_scoring_mlp
from hakken_models.models.segal.lightning import _global_to_local
from hakken_models.steps.dataset.build_relation_labels import (
    build_fact_relation_labels,
    build_pair_relation_history,
)

DEVICE = "cuda"
SEED = 42


# ── synthetic data helpers ────────────────────────────────────────────────────


def _make_embedder_config(embedding_dim: int) -> EmbedderConfig:
    return EmbedderConfig(
        provider="huggingface",
        model_name="test-model",
        embedding_dim=embedding_dim,
    )


def _make_segal_config(
    encoder_dim: int = 32,
    embedding_dim: int = 64,
    gnn_kwargs: dict[str, Any] | None = None,
    temporal_dim: int = 16,
    learnable_frequencies: bool = True,
    scoring_hidden: int = 32,
    scoring_layers: int = 2,
    scoring_dropout: float = 0.0,
    edge_feature_mode: str = "cat",
) -> SeGALConfig:
    if gnn_kwargs is None:
        gnn_kwargs = {"hidden_channels": 32, "num_layers": 2, "heads": 2, "v2": True}
    return SeGALConfig(
        encoder_dim=encoder_dim,
        embedder=_make_embedder_config(embedding_dim),
        gnn=GNNConfig(name="GAT", kwargs=gnn_kwargs),
        temporal=TemporalEncoderConfig(
            embedding_dim=temporal_dim,
            learnable_frequencies=learnable_frequencies,
        ),
        scoring=ScoringConfig(
            hidden_dim=scoring_hidden,
            num_layers=scoring_layers,
            dropout=scoring_dropout,
        ),
        edge_feature_mode=edge_feature_mode,
    )


def _make_context_graph(
    num_nodes: int = 20,
    num_edges: int = 50,
    num_relations: int = 5,
    embedding_dim: int = 64,
) -> tuple[KGData, Tensor]:
    """Build a synthetic context graph and relation embedding matrix on DEVICE."""
    torch.manual_seed(SEED)
    x = torch.randn(num_nodes, embedding_dim, device=DEVICE)
    edge_index = torch.randint(0, num_nodes, (2, num_edges), device=DEVICE)
    rel_ids = torch.randint(0, num_relations, (num_edges,), device=DEVICE).float()
    timestamps = (torch.rand(num_edges, device=DEVICE) * 35 + 1990).float()
    edge_attr = torch.stack([rel_ids, timestamps], dim=1)

    graph = KGData(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)
    rel_embs = torch.randn(num_relations, embedding_dim, device=DEVICE)
    return graph, rel_embs


def _make_lit_segal_batch(
    num_subgraph_nodes: int = 30,
    num_context_edges: int = 60,
    num_relations: int = 5,
    batch_size: int = 8,
    num_negatives: int = 4,
    with_relation_labels: bool = False,
) -> KGData:
    """Build a KGData batch as produced by TemporalKGLinkNeighborLoader on DEVICE."""
    torch.manual_seed(SEED)
    n_id = torch.arange(num_subgraph_nodes, device=DEVICE)

    edge_index = torch.randint(0, num_subgraph_nodes, (2, num_context_edges), device=DEVICE)
    rel_ids = torch.randint(0, num_relations, (num_context_edges,), device=DEVICE).float()
    timestamps = (torch.rand(num_context_edges, device=DEVICE) * 35 + 1990).float()
    edge_attr = torch.stack([rel_ids, timestamps], dim=1)

    label_s = n_id[torch.randint(0, num_subgraph_nodes, (batch_size,), device=DEVICE)]
    label_o = n_id[torch.randint(0, num_subgraph_nodes, (batch_size,), device=DEVICE)]
    edge_label_index = torch.stack([label_s, label_o], dim=0)
    edge_label = torch.randint(0, num_relations, (batch_size,), device=DEVICE)

    neg_edge_label_index = corrupt_entity_pairs(
        subjects=label_s,
        objects=label_o,
        n_id=n_id,
        num_negatives=num_negatives,
    )

    relation_labels = None
    if with_relation_labels:
        relation_labels = torch.zeros(batch_size, num_relations, device=DEVICE)
        for i in range(batch_size):
            relation_labels[i, edge_label[i]] = 1.0
            extra = torch.randint(0, num_relations, (2,))
            relation_labels[i, extra] = 1.0

    return KGData(
        x=None,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=num_subgraph_nodes,
        n_id=n_id,
        edge_label_index=edge_label_index,
        edge_label=edge_label,
        neg_edge_label_index=neg_edge_label_index,
        relation_labels=relation_labels,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  TemporalEncoder
# ═══════════════════════════════════════════════════════════════════════════════


class TestTemporalEncoder:
    __test__ = True

    @pytest.fixture(params=[16, 32, 64])
    def embedding_dim(self, request: pytest.FixtureRequest) -> int:
        return request.param

    def test_output_shape(self, embedding_dim: int) -> None:
        enc = TemporalEncoder(embedding_dim=embedding_dim).to(DEVICE)
        t = torch.tensor([1990.0, 2000.0, 2024.5], device=DEVICE)
        out = enc(t)
        assert out.shape == (3, embedding_dim)

    def test_output_shape_batched(self) -> None:
        enc = TemporalEncoder(embedding_dim=32).to(DEVICE)
        t = torch.randn(4, 10, device=DEVICE)
        out = enc(t)
        assert out.shape == (4, 10, 32)

    def test_set_normalization(self) -> None:
        enc = TemporalEncoder(embedding_dim=16).to(DEVICE)
        enc.set_normalization(mean=2000.0, std=15.0)
        assert enc.t_mean.item() == pytest.approx(2000.0)
        assert enc.t_std.item() == pytest.approx(15.0)

    def test_normalization_clamps_zero_std(self) -> None:
        enc = TemporalEncoder(embedding_dim=16).to(DEVICE)
        enc.set_normalization(mean=2000.0, std=0.0)
        assert enc.t_std.item() == pytest.approx(1e-6, abs=1e-9)

    def test_learnable_frequencies(self) -> None:
        enc = TemporalEncoder(embedding_dim=32, learnable_frequencies=True).to(DEVICE)
        assert isinstance(enc.frequencies, nn.Parameter)
        assert isinstance(enc.phases, nn.Parameter)

    def test_fixed_frequencies(self) -> None:
        enc = TemporalEncoder(embedding_dim=32, learnable_frequencies=False).to(DEVICE)
        assert not isinstance(enc.frequencies, nn.Parameter)
        assert not isinstance(enc.phases, nn.Parameter)

    def test_custom_num_sinusoidal(self) -> None:
        enc = TemporalEncoder(embedding_dim=32, num_sinusoidal=10).to(DEVICE)
        assert enc.frequencies.shape == (10,)
        assert enc.linear.in_features == 1
        assert enc.linear.out_features == 22  # 32 - 10

    def test_numeric_stability(self) -> None:
        enc = TemporalEncoder(embedding_dim=64).to(DEVICE)
        enc.set_normalization(mean=2000.0, std=15.0)
        t = torch.tensor([0.0, -1000.0, 5000.0, 2025.0], device=DEVICE)
        out = enc(t)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_gradient_flow(self) -> None:
        enc = TemporalEncoder(embedding_dim=32, learnable_frequencies=True).to(DEVICE)
        t = torch.tensor([1990.0, 2000.0, 2024.0], device=DEVICE)
        out = enc(t)
        out.sum().backward()
        for p in enc.parameters():
            if p.requires_grad:
                assert p.grad is not None


# ═══════════════════════════════════════════════════════════════════════════════
#  _build_scoring_mlp
# ═══════════════════════════════════════════════════════════════════════════════


class TestBuildScoringMLP:
    __test__ = True

    @pytest.mark.parametrize("num_layers", [1, 2, 3])
    def test_output_shape(self, num_layers: int) -> None:
        mlp = _build_scoring_mlp(in_dim=96, hidden_dim=64, num_layers=num_layers).to(DEVICE)
        x = torch.randn(8, 96, device=DEVICE)
        out = mlp(x)
        assert out.shape == (8, 1)

    def test_single_sample(self) -> None:
        mlp = _build_scoring_mlp(in_dim=48, hidden_dim=32, num_layers=2).to(DEVICE)
        x = torch.randn(1, 48, device=DEVICE)
        out = mlp(x)
        assert out.shape == (1, 1)

    def test_layer_structure(self) -> None:
        mlp = _build_scoring_mlp(in_dim=96, hidden_dim=64, num_layers=2, dropout=0.1)
        linear_layers = [m for m in mlp if isinstance(m, nn.Linear)]
        assert len(linear_layers) == 3  # 2 hidden + 1 output
        assert linear_layers[0].in_features == 96
        assert linear_layers[0].out_features == 64
        assert linear_layers[-1].out_features == 1


# ═══════════════════════════════════════════════════════════════════════════════
#  SeGALConfig
# ═══════════════════════════════════════════════════════════════════════════════


class TestSeGALConfig:
    __test__ = True

    def test_valid_construction(self) -> None:
        cfg = _make_segal_config()
        assert cfg.encoder_dim == 32
        assert cfg.edge_feature_mode == "cat"

    def test_defaults_applied(self) -> None:
        cfg = SeGALConfig(
            encoder_dim=64,
            embedder=_make_embedder_config(128),
        )
        assert cfg.temporal.embedding_dim == 64
        assert cfg.scoring.num_layers == 2
        assert cfg.edge_feature_mode == "cat"
        assert cfg.gnn.name == "GraphSAGE"

    def test_add_mode_config(self) -> None:
        cfg = _make_segal_config(encoder_dim=32, temporal_dim=32, edge_feature_mode="add")
        assert cfg.edge_feature_mode == "add"
        assert cfg.encoder_dim == cfg.temporal.embedding_dim


# ═══════════════════════════════════════════════════════════════════════════════
#  SeGAL model
# ═══════════════════════════════════════════════════════════════════════════════


SEGAL_CONFIGS: list[dict[str, Any]] = [
    {
        "encoder_dim": 32,
        "embedding_dim": 64,
        "temporal_dim": 16,
        "edge_feature_mode": "cat",
        "gnn_kwargs": {"hidden_channels": 32, "num_layers": 2, "heads": 2, "v2": True},
        "learnable_frequencies": True,
    },
    {
        "encoder_dim": 32,
        "embedding_dim": 32,
        "temporal_dim": 16,
        "edge_feature_mode": "cat",
        "gnn_kwargs": {"hidden_channels": 32, "num_layers": 1, "heads": 1, "v2": True},
        "learnable_frequencies": False,
    },
    {
        "encoder_dim": 32,
        "embedding_dim": 64,
        "temporal_dim": 32,
        "edge_feature_mode": "add",
        "gnn_kwargs": {
            "hidden_channels": 32,
            "num_layers": 2,
            "heads": 2,
            "v2": True,
            "dropout": 0.1,
        },
        "learnable_frequencies": True,
    },
]

NUM_NODES = 20
NUM_EDGES = 50
NUM_RELATIONS = 5


class TestSeGAL:
    __test__ = True

    @pytest.fixture(params=SEGAL_CONFIGS, ids=["cat-proj", "cat-identity", "add-proj"])
    def segal_cfg(self, request: pytest.FixtureRequest) -> dict[str, Any]:
        return request.param

    @pytest.fixture()
    def model(self, segal_cfg: dict[str, Any]) -> SeGAL:
        config = _make_segal_config(
            encoder_dim=segal_cfg["encoder_dim"],
            embedding_dim=segal_cfg["embedding_dim"],
            temporal_dim=segal_cfg["temporal_dim"],
            edge_feature_mode=segal_cfg["edge_feature_mode"],
            gnn_kwargs=segal_cfg["gnn_kwargs"],
            learnable_frequencies=segal_cfg["learnable_frequencies"],
        )
        return SeGAL(config).to(DEVICE)

    @pytest.fixture()
    def context_data(self, segal_cfg: dict[str, Any]) -> tuple[KGData, Tensor]:
        return _make_context_graph(
            num_nodes=NUM_NODES,
            num_edges=NUM_EDGES,
            num_relations=NUM_RELATIONS,
            embedding_dim=segal_cfg["embedding_dim"],
        )

    # ── initialisation ────────────────────────────────────────────────────

    def test_is_nn_module(self, model: SeGAL) -> None:
        assert isinstance(model, nn.Module)

    def test_has_expected_attributes(self, model: SeGAL) -> None:
        assert hasattr(model, "config")
        assert hasattr(model, "input_proj")
        assert hasattr(model, "temporal_encoder")
        assert hasattr(model, "gnn")
        assert hasattr(model, "scoring_mlp")

    def test_input_projection_type(self, model: SeGAL, segal_cfg: dict[str, Any]) -> None:
        assert isinstance(model.input_proj, nn.Sequential)
        assert len(model.input_proj) == 2
        assert isinstance(model.input_proj[0], nn.LayerNorm)
        if segal_cfg["embedding_dim"] == segal_cfg["encoder_dim"]:
            assert isinstance(model.input_proj[1], nn.Identity)
        else:
            assert isinstance(model.input_proj[1], nn.Linear)

    # ── _build_edge_features ──────────────────────────────────────────────

    def test_build_edge_features_shape(self, model: SeGAL, segal_cfg: dict[str, Any]) -> None:
        num_ctx_edges = 30
        enc_dim = segal_cfg["encoder_dim"]
        temp_dim = segal_cfg["temporal_dim"]

        rel_embs = torch.randn(num_ctx_edges, enc_dim, device=DEVICE)
        timestamps = torch.randn(num_ctx_edges, device=DEVICE) * 15 + 2000

        out = model._build_edge_features(rel_embs, timestamps)

        if segal_cfg["edge_feature_mode"] == "cat":
            assert out.shape == (num_ctx_edges, enc_dim + temp_dim)
        else:
            assert out.shape == (num_ctx_edges, enc_dim)

    # ── encode_context ────────────────────────────────────────────────────

    def test_encode_context_shape(
        self, model: SeGAL, context_data: tuple[KGData, Tensor], segal_cfg: dict[str, Any]
    ) -> None:
        graph, rel_embs = context_data
        out = model.encode_context(graph, rel_embs)
        assert out.shape == (NUM_NODES, segal_cfg["encoder_dim"])

    def test_encode_context_numeric_stability(
        self, model: SeGAL, context_data: tuple[KGData, Tensor]
    ) -> None:
        graph, rel_embs = context_data
        out = model.encode_context(graph, rel_embs)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    # ── score ─────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_score_shape(
        self,
        model: SeGAL,
        context_data: tuple[KGData, Tensor],
        segal_cfg: dict[str, Any],
        batch_size: int,
    ) -> None:
        graph, rel_embs = context_data
        enc_dim = segal_cfg["encoder_dim"]

        subject_idx = torch.randint(0, NUM_NODES, (batch_size,), device=DEVICE)
        target_idx = torch.randint(0, NUM_NODES, (batch_size,), device=DEVICE)
        relation_emb = torch.randn(batch_size, enc_dim, device=DEVICE)

        scores = model.score(subject_idx, relation_emb, target_idx, graph, rel_embs)
        assert scores.shape == (batch_size,)

    def test_score_numeric_stability(
        self, model: SeGAL, context_data: tuple[KGData, Tensor], segal_cfg: dict[str, Any]
    ) -> None:
        graph, rel_embs = context_data
        enc_dim = segal_cfg["encoder_dim"]
        batch_size = 8

        subject_idx = torch.randint(0, NUM_NODES, (batch_size,), device=DEVICE)
        target_idx = torch.randint(0, NUM_NODES, (batch_size,), device=DEVICE)
        relation_emb = torch.randn(batch_size, enc_dim, device=DEVICE)

        scores = model.score(subject_idx, relation_emb, target_idx, graph, rel_embs)
        assert not torch.isnan(scores).any()
        assert not torch.isinf(scores).any()

    # ── score_embeddings ──────────────────────────────────────────────────

    @pytest.mark.parametrize("batch_size", [1, 8])
    def test_score_embeddings_shape(
        self, model: SeGAL, segal_cfg: dict[str, Any], batch_size: int
    ) -> None:
        enc_dim = segal_cfg["encoder_dim"]
        s = torch.randn(batch_size, enc_dim, device=DEVICE)
        r = torch.randn(batch_size, enc_dim, device=DEVICE)
        o = torch.randn(batch_size, enc_dim, device=DEVICE)
        scores = model.score_embeddings(s, r, o)
        assert scores.shape == (batch_size,)

    # ── forward == score ──────────────────────────────────────────────────

    def test_forward_equals_score(
        self, model: SeGAL, context_data: tuple[KGData, Tensor], segal_cfg: dict[str, Any]
    ) -> None:
        graph, rel_embs = context_data
        enc_dim = segal_cfg["encoder_dim"]
        batch_size = 4

        torch.manual_seed(SEED)
        subject_idx = torch.randint(0, NUM_NODES, (batch_size,), device=DEVICE)
        target_idx = torch.randint(0, NUM_NODES, (batch_size,), device=DEVICE)
        relation_emb = torch.randn(batch_size, enc_dim, device=DEVICE)

        model.eval()
        with torch.no_grad():
            via_score = model.score(subject_idx, relation_emb, target_idx, graph, rel_embs)
            via_forward = model.forward(subject_idx, relation_emb, target_idx, graph, rel_embs)

        torch.testing.assert_close(via_score, via_forward)

    # ── gradient flow ─────────────────────────────────────────────────────

    def test_gradient_flow(
        self, model: SeGAL, context_data: tuple[KGData, Tensor], segal_cfg: dict[str, Any]
    ) -> None:
        graph, rel_embs = context_data
        enc_dim = segal_cfg["encoder_dim"]
        batch_size = 4

        subject_idx = torch.randint(0, NUM_NODES, (batch_size,), device=DEVICE)
        target_idx = torch.randint(0, NUM_NODES, (batch_size,), device=DEVICE)
        relation_emb = torch.randn(batch_size, enc_dim, device=DEVICE)

        scores = model.score(subject_idx, relation_emb, target_idx, graph, rel_embs)
        scores.sum().backward()

        has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        assert has_grad, "At least one parameter should have gradients"

    # ── deterministic eval ────────────────────────────────────────────────

    def test_deterministic_in_eval(
        self, model: SeGAL, context_data: tuple[KGData, Tensor], segal_cfg: dict[str, Any]
    ) -> None:
        """Two forward passes in eval mode must return identical results."""
        graph, rel_embs = context_data
        enc_dim = segal_cfg["encoder_dim"]
        batch_size = 4

        subject_idx = torch.randint(0, NUM_NODES, (batch_size,), device=DEVICE)
        target_idx = torch.randint(0, NUM_NODES, (batch_size,), device=DEVICE)
        relation_emb = torch.randn(batch_size, enc_dim, device=DEVICE)

        model.eval()
        with torch.no_grad():
            out1 = model(subject_idx, relation_emb, target_idx, graph, rel_embs)
            out2 = model(subject_idx, relation_emb, target_idx, graph, rel_embs)

        torch.testing.assert_close(out1, out2)

    # ── parameter count sanity ────────────────────────────────────────────

    def test_has_learnable_parameters(self, model: SeGAL) -> None:
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert total > 0

    # ── error handling ────────────────────────────────────────────────────

    def test_edge_mode_add_dim_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="encoder_dim must equal"):
            cfg = _make_segal_config(encoder_dim=32, temporal_dim=16, edge_feature_mode="add")
            SeGAL(cfg)

    def test_unknown_edge_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown edge_feature_mode"):
            cfg = _make_segal_config(edge_feature_mode="multiply")
            SeGAL(cfg)


# ═══════════════════════════════════════════════════════════════════════════════
#  SeGAL loss
# ═══════════════════════════════════════════════════════════════════════════════


class TestRankingRelationLoss:
    __test__ = True

    def test_entity_only_returns_scalar(self) -> None:
        loss_fn = RankingRelationLoss(
            entity_loss="MarginRankingLoss",
            entity_loss_kwargs={"margin": 1.0},
            relation_loss="BCEWithLogitsLoss",
            neg_strategy="hardest",
        ).to(DEVICE)
        pos = torch.tensor([1.0, 0.5, 0.8], device=DEVICE)
        neg = torch.tensor([[0.2, 0.3], [0.1, 0.6], [0.4, 0.9]], device=DEVICE)
        loss, _ = loss_fn(pos, neg)
        assert loss.dim() == 0
        assert not torch.isnan(loss)

    def test_entity_plus_relation_returns_scalar(self) -> None:
        loss_fn = RankingRelationLoss(
            entity_loss="MarginRankingLoss",
            entity_loss_kwargs={"margin": 1.0},
            relation_loss="BCEWithLogitsLoss",
            rel_loss_weight=0.5,
            neg_strategy="hardest",
        ).to(DEVICE)
        pos = torch.tensor([1.0, 0.5], device=DEVICE)
        neg = torch.tensor([[0.2, 0.3], [0.1, 0.6]], device=DEVICE)
        rel_logits = torch.randn(2, 5, device=DEVICE)
        rel_labels = torch.tensor([[1, 0, 1, 0, 0], [0, 1, 0, 0, 1]], device=DEVICE).float()
        loss, _ = loss_fn(pos, neg, rel_logits=rel_logits, rel_labels=rel_labels)
        assert loss.dim() == 0
        assert not torch.isnan(loss)

    def test_rel_loss_weight_scales_relation_term(self) -> None:
        loss_fn = RankingRelationLoss(
            entity_loss="MarginRankingLoss",
            entity_loss_kwargs={"margin": 1.0},
            relation_loss="BCEWithLogitsLoss",
            rel_loss_weight=2.0,
            neg_strategy="hardest",
        ).to(DEVICE)
        pos = torch.tensor([1.0], device=DEVICE)
        neg = torch.tensor([[0.2]], device=DEVICE)
        rel_logits = torch.randn(1, 3, device=DEVICE)
        rel_labels = torch.tensor([[1, 0, 1]], device=DEVICE).float()
        loss, _ = loss_fn(pos, neg, rel_logits=rel_logits, rel_labels=rel_labels)
        assert loss.dim() == 0

    def test_neg_strategy_mean(self) -> None:
        loss_fn = RankingRelationLoss(
            entity_loss="MarginRankingLoss",
            entity_loss_kwargs={"margin": 1.0},
            neg_strategy="mean",
        ).to(DEVICE)
        pos = torch.tensor([1.0, 0.5], device=DEVICE)
        neg = torch.tensor([[0.2, 0.3], [0.1, 0.6]], device=DEVICE)
        loss, _ = loss_fn(pos, neg)
        assert loss.dim() == 0

    def test_entity_loss_multi_neg_nssa(self) -> None:
        loss_fn = RankingRelationLoss(
            entity_loss="NSSALoss",
            entity_loss_kwargs={"margin": 9.0},
            neg_strategy="hardest",
            entity_loss_multi_neg=True,
        ).to(DEVICE)
        pos = torch.tensor([1.0, 0.5], device=DEVICE)
        neg = torch.tensor([[0.2, 0.3], [0.1, 0.6]], device=DEVICE)
        loss, _ = loss_fn(pos, neg)
        assert loss.dim() == 0

    def test_focal_loss_for_relations(self) -> None:
        loss_fn = RankingRelationLoss(
            entity_loss="MarginRankingLoss",
            entity_loss_kwargs={"margin": 1.0},
            relation_loss="FLWithLogitsLoss",
            relation_loss_kwargs={"gamma": 2.0},
            rel_loss_weight=1.0,
            neg_strategy="hardest",
        ).to(DEVICE)
        pos = torch.tensor([1.0], device=DEVICE)
        neg = torch.tensor([[0.2]], device=DEVICE)
        rel_logits = torch.randn(1, 4, device=DEVICE)
        rel_labels = torch.tensor([[1, 0, 1, 0]], device=DEVICE).float()
        loss, _ = loss_fn(pos, neg, rel_logits=rel_logits, rel_labels=rel_labels)
        assert loss.dim() == 0

    def test_asymmetric_loss_for_relations(self) -> None:
        loss_fn = RankingRelationLoss(
            entity_loss="MarginRankingLoss",
            entity_loss_kwargs={"margin": 1.0},
            relation_loss="AsymmetricLoss",
            relation_loss_kwargs={"gamma_neg": 4.0, "gamma_pos": 1.0, "clip": 0.05},
            rel_loss_weight=1.0,
            neg_strategy="hardest",
        ).to(DEVICE)
        pos = torch.tensor([1.0], device=DEVICE)
        neg = torch.tensor([[0.2]], device=DEVICE)
        rel_logits = torch.randn(1, 4, device=DEVICE)
        rel_labels = torch.tensor([[1, 0, 1, 0]], device=DEVICE).float()
        loss, _ = loss_fn(pos, neg, rel_logits=rel_logits, rel_labels=rel_labels)
        assert loss.dim() == 0

    def test_compute_pos_weight_from_relation_labels(self) -> None:
        labels = torch.tensor(
            [[1, 0, 1], [1, 0, 0], [0, 1, 1], [0, 0, 1]],
            dtype=torch.float32,
        )
        pw = compute_pos_weight_from_relation_labels(labels)
        assert pw.shape == (3,)
        assert pw[0] < pw[1]


# ═══════════════════════════════════════════════════════════════════════════════
#  Lightning helpers
# ═══════════════════════════════════════════════════════════════════════════════


class TestLitSeGALHelpers:
    __test__ = True

    def test_global_to_local(self) -> None:
        n_id = torch.tensor([10, 20, 30, 40, 50], device=DEVICE)
        global_ids = torch.tensor([30, 10, 50], device=DEVICE)
        local = _global_to_local(global_ids, n_id)
        assert (n_id[local] == global_ids).all()

    def test_global_to_local_single(self) -> None:
        n_id = torch.tensor([5, 15, 25], device=DEVICE)
        global_ids = torch.tensor([15], device=DEVICE)
        local = _global_to_local(global_ids, n_id)
        assert n_id[local].item() == 15

    def test_corrupt_entity_pairs_shape(self) -> None:
        n_id = torch.arange(50, device=DEVICE)
        subjects = torch.tensor([0, 5, 10, 20], device=DEVICE)
        objects = torch.tensor([1, 6, 11, 21], device=DEVICE)
        num_negatives = 8

        result = corrupt_entity_pairs(subjects, objects, n_id, num_negatives)
        assert result.shape == (2, 4, num_negatives)

    def test_corrupt_entity_pairs_values_in_n_id(self) -> None:
        n_id = torch.tensor([10, 20, 30, 40, 50], device=DEVICE)
        subjects = torch.tensor([10, 20], device=DEVICE)
        objects = torch.tensor([30, 40], device=DEVICE)

        result = corrupt_entity_pairs(subjects, objects, n_id, num_negatives=16)
        neg_s, neg_o = result[0], result[1]

        n_id_set = set(n_id.tolist())
        assert all(v in n_id_set for v in neg_s.flatten().tolist())
        assert all(v in n_id_set for v in neg_o.flatten().tolist())

    def test_corrupt_entity_pairs_preserves_one_side(self) -> None:
        """For each negative, either subject or object must be the original."""
        torch.manual_seed(SEED)
        n_id = torch.arange(100, device=DEVICE)
        subjects = torch.tensor([0, 1, 2, 3], device=DEVICE)
        objects = torch.tensor([10, 11, 12, 13], device=DEVICE)

        result = corrupt_entity_pairs(subjects, objects, n_id, num_negatives=64)
        neg_s, neg_o = result[0], result[1]

        s_kept = neg_s == subjects.unsqueeze(1)
        o_kept = neg_o == objects.unsqueeze(1)
        assert (s_kept | o_kept).all(), "Each negative must preserve either subject or object"

    # ── build_fact_relation_labels tests ──────────────────────────────────

    def test_build_fact_relation_labels_shape(self) -> None:
        facts = torch.tensor([[0, 1, 2, 5], [3, 0, 4, 6]])
        history = build_pair_relation_history(facts)
        labels = build_fact_relation_labels(facts, history, num_relations=3)
        assert labels.shape == (2, 3)

    def test_build_fact_relation_labels_includes_target(self) -> None:
        """The target relation r_i must always be set to 1."""
        facts = torch.tensor([[0, 2, 1, 5]])
        history = build_pair_relation_history(facts)
        labels = build_fact_relation_labels(facts, history, num_relations=5)
        assert labels[0, 2] == 1.0

    def test_build_fact_relation_labels_includes_prior_relations(self) -> None:
        """Relations observed for (s,o) at t' < t should be marked."""
        knowledge = torch.tensor(
            [
                [0, 0, 1, 1],  # (0,1) has r=0 at t=1
                [0, 1, 1, 2],  # (0,1) has r=1 at t=2
                [0, 3, 1, 5],  # target: (0,1) r=3 at t=5
            ]
        )
        target = knowledge[2:3]
        history = build_pair_relation_history(knowledge)
        labels = build_fact_relation_labels(target, history, num_relations=5)
        assert labels[0, 3] == 1.0  # target relation
        assert labels[0, 0] == 1.0  # r=0 at t=1 < 5
        assert labels[0, 1] == 1.0  # r=1 at t=2 < 5
        assert labels[0, 2] == 0.0  # never observed
        assert labels[0, 4] == 0.0  # never observed

    def test_build_fact_relation_labels_respects_temporal_cutoff(self) -> None:
        """Relations at t >= t_target must NOT be included."""
        knowledge = torch.tensor(
            [
                [0, 0, 1, 1],  # r=0 at t=1
                [0, 1, 1, 5],  # r=1 at t=5 (not strictly before target)
                [0, 2, 1, 3],  # target: r=2 at t=3
            ]
        )
        target = knowledge[2:3]
        history = build_pair_relation_history(knowledge)
        labels = build_fact_relation_labels(target, history, num_relations=3)
        assert labels[0, 2] == 1.0  # target
        assert labels[0, 0] == 1.0  # t=1 < 3
        assert labels[0, 1] == 0.0  # t=5 NOT < 3

    def test_build_fact_relation_labels_different_pairs(self) -> None:
        """History from unrelated pairs must not leak."""
        knowledge = torch.tensor(
            [
                [0, 0, 1, 1],  # (0,1) r=0
                [2, 1, 3, 1],  # (2,3) r=1
                [0, 2, 1, 5],  # target: (0,1) r=2 at t=5
            ]
        )
        target = knowledge[2:3]
        history = build_pair_relation_history(knowledge)
        labels = build_fact_relation_labels(target, history, num_relations=3)
        assert labels[0, 0] == 1.0  # same pair (0,1), t<5
        assert labels[0, 1] == 0.0  # different pair (2,3)
        assert labels[0, 2] == 1.0  # target

    def test_build_pair_relation_history(self) -> None:
        facts = torch.tensor([[0, 3, 1, 1], [0, 4, 1, 2], [1, 5, 2, 3]])
        history = build_pair_relation_history(facts)
        assert (0, 1) in history
        assert len(history[(0, 1)]) == 2
        assert history[(0, 1)] == [(3, 1.0), (4, 2.0)]
        assert history[(1, 2)] == [(5, 3.0)]


# ═══════════════════════════════════════════════════════════════════════════════
#  LitSeGAL
# ═══════════════════════════════════════════════════════════════════════════════


class TestLitSeGAL:
    __test__ = True

    NUM_TOTAL_NODES = 50
    NUM_RELATIONS = 5
    EMBEDDING_DIM = 32
    ENCODER_DIM = 32

    @pytest.fixture()
    def segal_model(self) -> SeGAL:
        cfg = _make_segal_config(
            encoder_dim=self.ENCODER_DIM,
            embedding_dim=self.EMBEDDING_DIM,
            temporal_dim=16,
            scoring_dropout=0.0,
        )
        return SeGAL(cfg).to(DEVICE)

    @pytest.fixture()
    def node_embeddings(self) -> Tensor:
        return torch.randn(self.NUM_TOTAL_NODES, self.EMBEDDING_DIM, device=DEVICE)

    @pytest.fixture()
    def relation_embeddings(self) -> Tensor:
        return torch.randn(self.NUM_RELATIONS, self.EMBEDDING_DIM, device=DEVICE)

    @pytest.fixture()
    def lit_model(
        self, segal_model: SeGAL, node_embeddings: Tensor, relation_embeddings: Tensor
    ) -> LitSeGAL:
        loss_fn = RankingRelationLoss(
            entity_loss="MarginRankingLoss",
            entity_loss_kwargs={"margin": 1.0},
            neg_strategy="hardest",
        )
        return LitSeGAL(
            segal=segal_model,
            node_embeddings=node_embeddings,
            relation_embeddings=relation_embeddings,
            loss_fn=loss_fn,
        ).to(DEVICE)

    @pytest.fixture()
    def batch(self) -> KGData:
        return _make_lit_segal_batch(
            num_subgraph_nodes=self.NUM_TOTAL_NODES,
            num_context_edges=80,
            num_relations=self.NUM_RELATIONS,
            batch_size=8,
        )

    # ── initialisation ────────────────────────────────────────────────────

    def test_is_lightning_module(self, lit_model: LitSeGAL) -> None:
        from lightning.pytorch import LightningModule

        assert isinstance(lit_model, LightningModule)

    def test_buffers_registered(self, lit_model: LitSeGAL) -> None:
        node_t, rel_t = lit_model.embedding_tables()
        assert node_t.shape == (self.NUM_TOTAL_NODES, self.EMBEDDING_DIM)
        assert rel_t.shape == (self.NUM_RELATIONS, self.EMBEDDING_DIM)
        assert hasattr(lit_model, "node_embs")
        assert hasattr(lit_model, "rel_embs")

    # ── _prepare_batch ────────────────────────────────────────────────────

    def test_prepare_batch_injects_embeddings(self, lit_model: LitSeGAL, batch: KGData) -> None:
        assert batch.x is None
        prepared = lit_model._prepare_batch(batch)
        assert prepared.x is not None
        assert prepared.x.shape == (self.NUM_TOTAL_NODES, self.EMBEDDING_DIM)

    # ── training_step ─────────────────────────────────────────────────────

    def test_training_step_returns_scalar(self, lit_model: LitSeGAL, batch: KGData) -> None:
        loss = lit_model.training_step(batch, 0)
        assert loss.dim() == 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_training_step_gradient_flow(self, lit_model: LitSeGAL, batch: KGData) -> None:
        loss = lit_model.training_step(batch, 0)
        loss.backward()
        has_grad = any(p.grad is not None for p in lit_model.segal.parameters() if p.requires_grad)
        assert has_grad

    def test_learn_embeddings_embedding_gradients_after_step(self) -> None:
        cfg = _make_segal_config(
            encoder_dim=self.ENCODER_DIM,
            embedding_dim=self.EMBEDDING_DIM,
            temporal_dim=16,
            scoring_dropout=0.0,
        )
        cfg_learn = cfg.model_copy(update={"learn_embeddings": True})
        segal_learn = SeGAL(cfg_learn).to(DEVICE)
        node_e = torch.randn(self.NUM_TOTAL_NODES, self.ENCODER_DIM, device=DEVICE)
        rel_e = torch.randn(self.NUM_RELATIONS, self.ENCODER_DIM, device=DEVICE)
        lit = LitSeGAL(
            segal=segal_learn,
            node_embeddings=node_e,
            relation_embeddings=rel_e,
            loss_fn=RankingRelationLoss(
                entity_loss="MarginRankingLoss",
                entity_loss_kwargs={"margin": 1.0},
                neg_strategy="hardest",
            ),
            learn_embeddings=True,
        ).to(DEVICE)
        batch = _make_lit_segal_batch(
            num_subgraph_nodes=self.NUM_TOTAL_NODES,
            num_context_edges=80,
            num_relations=self.NUM_RELATIONS,
            batch_size=8,
        )
        loss = lit.training_step(batch, 0)
        loss.backward()
        assert lit.node_emb.weight.grad is not None
        assert lit.rel_emb.weight.grad is not None
        assert lit.node_emb.weight.grad.abs().sum() > 0

    # ── training/validation with relation labels ─────────────────────────

    @pytest.fixture()
    def batch_with_relation_labels(self) -> KGData:
        return _make_lit_segal_batch(
            num_subgraph_nodes=self.NUM_TOTAL_NODES,
            num_context_edges=80,
            num_relations=self.NUM_RELATIONS,
            batch_size=8,
            with_relation_labels=True,
        )

    def test_training_step_with_relation_labels(
        self, lit_model: LitSeGAL, batch_with_relation_labels: KGData
    ) -> None:
        loss = lit_model.training_step(batch_with_relation_labels, 0)
        assert loss.dim() == 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_score_step_returns_rel_logits(
        self, lit_model: LitSeGAL, batch_with_relation_labels: KGData
    ) -> None:
        batch_with_relation_labels = lit_model._prepare_batch(batch_with_relation_labels)
        out = lit_model._score_step(batch_with_relation_labels)
        assert out.rel_logits is not None
        assert out.rel_labels is not None
        assert out.rel_logits.shape == out.rel_labels.shape

    def test_gradient_flow_with_relation_labels(
        self, lit_model: LitSeGAL, batch_with_relation_labels: KGData
    ) -> None:
        loss = lit_model.training_step(batch_with_relation_labels, 0)
        loss.backward()
        has_grad = any(p.grad is not None for p in lit_model.segal.parameters() if p.requires_grad)
        assert has_grad

    def test_training_step_with_segal_composite_loss(
        self, segal_model: SeGAL, node_embeddings: Tensor, relation_embeddings: Tensor
    ) -> None:
        loss_fn = RankingRelationLoss(
            entity_loss="MarginRankingLoss",
            entity_loss_kwargs={"margin": 1.0},
            relation_loss="BCEWithLogitsLoss",
            neg_strategy="hardest",
        )
        lit = LitSeGAL(
            segal=segal_model,
            node_embeddings=node_embeddings,
            relation_embeddings=relation_embeddings,
            loss_fn=loss_fn,
        ).to(DEVICE)
        batch = _make_lit_segal_batch(
            num_subgraph_nodes=self.NUM_TOTAL_NODES,
            num_context_edges=80,
            num_relations=self.NUM_RELATIONS,
            batch_size=8,
        )
        loss = lit.training_step(batch, 0)
        assert loss.dim() == 0
        assert not torch.isnan(loss)

    # ── validation_step ───────────────────────────────────────────────────

    def test_validation_step_returns_scalar(self, lit_model: LitSeGAL, batch: KGData) -> None:
        loss = lit_model.validation_step(batch, 0)
        assert loss.dim() == 0
        assert not torch.isnan(loss)

    def test_validation_step_without_val_metric_hub(
        self,
        segal_model: SeGAL,
        node_embeddings: Tensor,
        relation_embeddings: Tensor,
        batch: KGData,
    ) -> None:
        lit = LitSeGAL(
            segal=segal_model,
            node_embeddings=node_embeddings,
            relation_embeddings=relation_embeddings,
            loss_fn=RankingRelationLoss(
                entity_loss="MarginRankingLoss",
                entity_loss_kwargs={"margin": 1.0},
                neg_strategy="hardest",
            ),
            val_metric_hub=None,
        ).to(DEVICE)
        loss = lit.validation_step(batch, 0)
        assert loss.dim() == 0

    def test_val_metric_hub_accumulates_across_validation_steps(
        self, lit_model: LitSeGAL, batch: KGData
    ) -> None:
        n_e, r_e = lit_model.embedding_tables()
        lit = LitSeGAL(
            segal=lit_model.segal,
            node_embeddings=n_e.detach().clone(),
            relation_embeddings=r_e.detach().clone(),
            loss_fn=lit_model.loss_fn,
            val_metric_hub=build_default_lit_kge_val_metric_hub(),
        ).to(DEVICE)
        lit.on_validation_epoch_start()
        lit.validation_step(batch, 0)
        lit.validation_step(batch, 0)
        assert lit.val_metric_hub is not None
        mr = lit.val_metric_hub.compute()["mean_rank"]
        assert torch.isfinite(mr)

    def test_on_validation_epoch_end_resets_val_metric_hub(
        self, lit_model: LitSeGAL, batch: KGData
    ) -> None:
        n_e, r_e = lit_model.embedding_tables()
        lit = LitSeGAL(
            segal=lit_model.segal,
            node_embeddings=n_e.detach().clone(),
            relation_embeddings=r_e.detach().clone(),
            loss_fn=lit_model.loss_fn,
            val_metric_hub=build_default_lit_kge_val_metric_hub(),
        ).to(DEVICE)
        lit.on_validation_epoch_start()
        lit.validation_step(batch, 0)
        lit.on_validation_epoch_end()
        assert lit.val_metric_hub is not None
        assert math.isinf(lit.val_metric_hub.compute()["mean_rank"].item())

    # ── compute_loss ──────────────────────────────────────────────────────

    def test_compute_loss_hardest(self, lit_model: LitSeGAL) -> None:
        pos = torch.tensor([1.0, 0.5, 0.8], device=DEVICE)
        neg = torch.tensor([[0.2, 0.3], [0.1, 0.6], [0.4, 0.9]], device=DEVICE)
        loss, _ = lit_model.compute_loss(pos, neg)
        assert loss.dim() == 0

    def test_compute_loss_mean_strategy(
        self, segal_model: SeGAL, node_embeddings: Tensor, relation_embeddings: Tensor
    ) -> None:
        lit = LitSeGAL(
            segal=segal_model,
            node_embeddings=node_embeddings,
            relation_embeddings=relation_embeddings,
            loss_fn=RankingRelationLoss(
                entity_loss="MarginRankingLoss",
                entity_loss_kwargs={"margin": 1.0},
                neg_strategy="mean",
            ),
        ).to(DEVICE)
        pos = torch.tensor([1.0, 0.5], device=DEVICE)
        neg = torch.tensor([[0.2, 0.3], [0.1, 0.6]], device=DEVICE)
        loss, _ = lit.compute_loss(pos, neg)
        assert loss.dim() == 0

    def test_compute_loss_single_negative(self, lit_model: LitSeGAL) -> None:
        pos = torch.tensor([1.0, 0.5], device=DEVICE)
        neg = torch.tensor([[0.2], [0.3]], device=DEVICE)
        loss, _ = lit_model.compute_loss(pos, neg)
        assert loss.dim() == 0

    def test_unknown_neg_strategy_raises(
        self, segal_model: SeGAL, node_embeddings: Tensor, relation_embeddings: Tensor
    ) -> None:
        lit = LitSeGAL(
            segal=segal_model,
            node_embeddings=node_embeddings,
            relation_embeddings=relation_embeddings,
            loss_fn=RankingRelationLoss(
                entity_loss="MarginRankingLoss",
                entity_loss_kwargs={},
                neg_strategy="unknown",
            ),
        ).to(DEVICE)
        with pytest.raises(NotImplementedError, match="Unknown neg_strategy"):
            lit.compute_loss(
                torch.tensor([1.0], device=DEVICE),
                torch.tensor([[0.2, 0.3]], device=DEVICE),
            )

    # ── configure_optimizers ──────────────────────────────────────────────

    def test_configure_optimizers_no_scheduler(self, lit_model: LitSeGAL) -> None:
        opt = lit_model.configure_optimizers()
        assert isinstance(opt, torch.optim.Adam)

    def test_configure_optimizers_with_scheduler(
        self, segal_model: SeGAL, node_embeddings: Tensor, relation_embeddings: Tensor
    ) -> None:
        lit = LitSeGAL(
            segal=segal_model,
            node_embeddings=node_embeddings,
            relation_embeddings=relation_embeddings,
            loss_fn=RankingRelationLoss(
                entity_loss="MarginRankingLoss",
                entity_loss_kwargs={},
                neg_strategy="hardest",
            ),
            scheduler_cls=torch.optim.lr_scheduler.StepLR,
            scheduler_kwargs={"step_size": 5},
        ).to(DEVICE)
        result = lit.configure_optimizers()
        assert isinstance(result, dict)
        assert "optimizer" in result
        assert "lr_scheduler" in result

    # ── _compute_mean_rank ────────────────────────────────────────────────

    def test_compute_mean_rank(self, lit_model: LitSeGAL) -> None:
        pos = torch.tensor([0.8, 0.3], device=DEVICE)
        neg = torch.tensor([[0.1, 0.5, 0.9], [0.1, 0.2, 0.4]], device=DEVICE)
        rank = lit_model._compute_mean_rank(pos, neg)
        assert rank.dim() == 0
        assert rank.item() >= 1.0

    def test_compute_mean_rank_returns_inf_on_nan(self, lit_model: LitSeGAL) -> None:
        """NaN scores must return inf so early stopping treats broken model as worst-case."""
        pos = torch.tensor([float("nan"), 0.3], device=DEVICE)
        neg = torch.tensor([[0.1, 0.5], [0.1, 0.2]], device=DEVICE)
        rank = lit_model._compute_mean_rank(pos, neg)
        assert rank.dim() == 0
        assert rank.item() == float("inf")


# ═══════════════════════════════════════════════════════════════════════════════
#  create_lit_segal factory
# ═══════════════════════════════════════════════════════════════════════════════


class TestCreateLitSegal:
    __test__ = True

    def test_factory_returns_lit_segal(self) -> None:
        cfg = _make_segal_config(encoder_dim=32, embedding_dim=32)
        model = SeGAL(cfg).to(DEVICE)
        node_embs = torch.randn(50, 32, device=DEVICE)
        rel_embs = torch.randn(5, 32, device=DEVICE)

        lit = create_lit_segal(
            segal=model,
            node_embeddings=node_embs,
            relation_embeddings=rel_embs,
            loss_config={"name": "MarginRankingLoss", "kwargs": {"margin": 1.0}},
            optimizer_config={"name": "Adam", "kwargs": {"lr": 1e-3}},
        )
        assert isinstance(lit, LitSeGAL)

    def test_factory_with_scheduler(self) -> None:
        cfg = _make_segal_config(encoder_dim=32, embedding_dim=32)
        model = SeGAL(cfg).to(DEVICE)
        node_embs = torch.randn(50, 32, device=DEVICE)
        rel_embs = torch.randn(5, 32, device=DEVICE)

        lit = create_lit_segal(
            segal=model,
            node_embeddings=node_embs,
            relation_embeddings=rel_embs,
            loss_config={"name": "MarginRankingLoss", "kwargs": {"margin": 1.0}},
            optimizer_config={"name": "Adam", "kwargs": {"lr": 1e-3}},
            scheduler_config={"name": "StepLR", "kwargs": {"step_size": 10}},
        )
        result = lit.configure_optimizers()
        assert isinstance(result, dict)
        assert "lr_scheduler" in result

    def test_factory_with_segal_composite_loss(self) -> None:
        cfg = _make_segal_config(encoder_dim=32, embedding_dim=32)
        model = SeGAL(cfg).to(DEVICE)
        node_embs = torch.randn(50, 32, device=DEVICE)
        rel_embs = torch.randn(5, 32, device=DEVICE)

        lit = create_lit_segal(
            segal=model,
            node_embeddings=node_embs,
            relation_embeddings=rel_embs,
            loss_config={
                "name": "RankingRelationLoss",
                "kwargs": {
                    "entity_loss": "MarginRankingLoss",
                    "entity_loss_kwargs": {"margin": 1.0},
                    "relation_loss": "BCEWithLogitsLoss",
                    "relation_loss_kwargs": {},
                    "neg_strategy": "hardest",
                    "rel_loss_weight": 1.0,
                },
            },
            optimizer_config={"name": "Adam", "kwargs": {"lr": 1e-3}},
        )
        assert isinstance(lit.loss_fn, RankingRelationLoss)


# ═══════════════════════════════════════════════════════════════════════════════
#  SeGALDataModule
# ═══════════════════════════════════════════════════════════════════════════════


def _make_synthetic_facts(
    num_facts: int,
    num_nodes: int,
    num_relations: int,
    num_timestamps: int,
    seed: int = SEED,
) -> Tensor:
    """Build a ``[num_facts, 4]`` facts tensor ``(s, r, o, t)`` on CPU."""
    torch.manual_seed(seed)
    subjects = torch.randint(0, num_nodes, (num_facts,))
    relations = torch.randint(0, num_relations, (num_facts,))
    objects = torch.randint(0, num_nodes, (num_facts,))
    timestamps = torch.randint(0, num_timestamps, (num_facts,))
    return torch.stack([subjects, relations, objects, timestamps], dim=1)


class TestSeGALDataModule:
    __test__ = True

    NUM_NODES = 50
    NUM_RELATIONS = 5
    NUM_TIMESTAMPS = 10
    NUM_TRAIN_FACTS = 200
    NUM_VAL_FACTS = 60
    BATCH_SIZE = 8
    NUM_NEIGHBORS: ClassVar[list[int]] = [5, 3]

    @pytest.fixture()
    def train_facts(self) -> Tensor:
        return _make_synthetic_facts(
            self.NUM_TRAIN_FACTS, self.NUM_NODES, self.NUM_RELATIONS, self.NUM_TIMESTAMPS, seed=1
        )

    @pytest.fixture()
    def val_facts(self) -> Tensor:
        return _make_synthetic_facts(
            self.NUM_VAL_FACTS, self.NUM_NODES, self.NUM_RELATIONS, self.NUM_TIMESTAMPS, seed=2
        )

    @pytest.fixture()
    def data_module(self, train_facts: Tensor, val_facts: Tensor) -> SeGALDataModule:
        return SeGALDataModule(
            train_facts=train_facts,
            val_facts=val_facts,
            num_nodes=self.NUM_NODES,
            num_neighbors=self.NUM_NEIGHBORS,
            batch_size=self.BATCH_SIZE,
        )

    # ── construction & internal state ─────────────────────────────────────

    def test_construction(self, data_module: SeGALDataModule) -> None:
        assert isinstance(data_module, SeGALDataModule)
        assert data_module.batch_size == self.BATCH_SIZE
        assert data_module.num_neighbors == self.NUM_NEIGHBORS

    def test_train_kg_data_is_valid(self, data_module: SeGALDataModule) -> None:
        assert_is_kg_data(data_module.train_kg_data)

    def test_val_kg_data_is_valid(self, data_module: SeGALDataModule) -> None:
        assert_is_kg_data(data_module.val_kg_data)

    def test_train_kg_data_edge_count(
        self, data_module: SeGALDataModule, train_facts: Tensor
    ) -> None:
        # Default add_reverse_edges=True: one forward + one reverse edge per fact
        assert data_module.train_kg_data.edge_index.shape[1] == 2 * train_facts.shape[0]

    def test_val_kg_data_includes_all_facts(
        self, data_module: SeGALDataModule, train_facts: Tensor, val_facts: Tensor
    ) -> None:
        n = train_facts.shape[0] + val_facts.shape[0]
        assert data_module.val_kg_data.edge_index.shape[1] == 2 * n

    def test_kg_data_edge_count_without_reverse_edges(
        self, train_facts: Tensor, val_facts: Tensor
    ) -> None:
        dm = SeGALDataModule(
            train_facts=train_facts,
            val_facts=val_facts,
            num_nodes=self.NUM_NODES,
            num_neighbors=self.NUM_NEIGHBORS,
            batch_size=self.BATCH_SIZE,
            add_reverse_edges=False,
        )
        assert dm.train_kg_data.edge_index.shape[1] == train_facts.shape[0]
        assert dm.val_kg_data.edge_index.shape[1] == train_facts.shape[0] + val_facts.shape[0]

    def test_val_kg_data_has_more_edges_than_train(self, data_module: SeGALDataModule) -> None:
        train_edges = data_module.train_kg_data.edge_index.shape[1]
        val_edges = data_module.val_kg_data.edge_index.shape[1]
        assert val_edges > train_edges

    # ── column extraction ─────────────────────────────────────────────────

    def test_train_entity_pairs_shape(
        self,
        data_module: SeGALDataModule,
        train_facts: Tensor,  # noqa: ARG002
    ) -> None:
        assert data_module.train_entity_pairs.shape == (self.NUM_TRAIN_FACTS, 2)

    def test_train_entity_pairs_values(
        self, data_module: SeGALDataModule, train_facts: Tensor
    ) -> None:
        expected = train_facts[:, [0, 2]]
        torch.testing.assert_close(data_module.train_entity_pairs, expected)

    def test_train_relations_shape(self, data_module: SeGALDataModule) -> None:
        assert data_module.train_relations.shape == (self.NUM_TRAIN_FACTS,)

    def test_train_relations_values(
        self, data_module: SeGALDataModule, train_facts: Tensor
    ) -> None:
        torch.testing.assert_close(data_module.train_relations, train_facts[:, 1])

    def test_train_timestamps_shape_and_dtype(self, data_module: SeGALDataModule) -> None:
        assert data_module.train_timestamps.shape == (self.NUM_TRAIN_FACTS,)
        assert data_module.train_timestamps.dtype == torch.float32

    def test_train_timestamps_values(
        self, data_module: SeGALDataModule, train_facts: Tensor
    ) -> None:
        torch.testing.assert_close(data_module.train_timestamps, train_facts[:, 3].float())

    def test_val_entity_pairs_shape(self, data_module: SeGALDataModule) -> None:
        assert data_module.val_entity_pairs.shape == (self.NUM_VAL_FACTS, 2)

    def test_val_entity_pairs_values(self, data_module: SeGALDataModule, val_facts: Tensor) -> None:
        expected = val_facts[:, [0, 2]]
        torch.testing.assert_close(data_module.val_entity_pairs, expected)

    def test_val_relations_values(self, data_module: SeGALDataModule, val_facts: Tensor) -> None:
        torch.testing.assert_close(data_module.val_relations, val_facts[:, 1])

    def test_val_timestamps_values(self, data_module: SeGALDataModule, val_facts: Tensor) -> None:
        torch.testing.assert_close(data_module.val_timestamps, val_facts[:, 3].float())

    # ── batch_size mutability (used by pipeline for auto-tuning) ──────────

    def test_batch_size_mutable(self, data_module: SeGALDataModule) -> None:
        data_module.batch_size = 64
        assert data_module.batch_size == 64

    def test_mutated_batch_size_propagates_to_dataloader(
        self, data_module: SeGALDataModule
    ) -> None:
        data_module.batch_size = 4
        loader = data_module.train_dataloader()
        batch = next(iter(loader))
        assert batch.edge_label.shape[0] == 4

    # ── dataloader factory methods ────────────────────────────────────────

    def test_train_dataloader_returns_iterable(self, data_module: SeGALDataModule) -> None:
        loader = data_module.train_dataloader()
        assert hasattr(loader, "__iter__")

    def test_val_dataloader_returns_iterable(self, data_module: SeGALDataModule) -> None:
        loader = data_module.val_dataloader()
        assert hasattr(loader, "__iter__")

    def test_train_batch_is_kg_data(self, data_module: SeGALDataModule) -> None:
        batch = next(iter(data_module.train_dataloader()))
        assert isinstance(batch, KGData)

    def test_val_batch_is_kg_data(self, data_module: SeGALDataModule) -> None:
        batch = next(iter(data_module.val_dataloader()))
        assert isinstance(batch, KGData)

    # ── batch structure (fields required by LitSeGAL) ─────────────────────

    def _assert_batch_has_lit_segal_fields(self, batch: KGData) -> None:
        """Verify a batch has all fields that LitSeGAL._score_step expects."""
        assert batch.n_id is not None, "batch.n_id required for _prepare_batch"
        assert batch.edge_index is not None, "batch.edge_index required for GNN"
        assert batch.edge_attr is not None, "batch.edge_attr required for edge features"
        assert batch.edge_label_index is not None, "batch.edge_label_index required for targets"
        assert batch.edge_label is not None, "batch.edge_label required for relation indices"
        assert batch.edge_label_index.shape[0] == 2
        assert batch.edge_attr.shape[1] >= 2
        assert batch.neg_edge_label_index is not None, (
            "batch.neg_edge_label_index required for negatives"
        )
        assert batch.neg_edge_label_index.shape[0] == 2

    def test_train_batch_has_lit_segal_fields(self, data_module: SeGALDataModule) -> None:
        batch = next(iter(data_module.train_dataloader()))
        self._assert_batch_has_lit_segal_fields(batch)

    def test_val_batch_has_lit_segal_fields(self, data_module: SeGALDataModule) -> None:
        batch = next(iter(data_module.val_dataloader()))
        self._assert_batch_has_lit_segal_fields(batch)

    def test_train_batch_edge_label_index_uses_global_ids(
        self, data_module: SeGALDataModule
    ) -> None:
        batch = next(iter(data_module.train_dataloader()))
        assert torch.all(torch.isin(batch.edge_label_index[0], batch.n_id))
        assert torch.all(torch.isin(batch.edge_label_index[1], batch.n_id))

    def test_train_covers_all_seed_edges(self, data_module: SeGALDataModule) -> None:
        total = sum(b.edge_label.shape[0] for b in data_module.train_dataloader())
        assert total == self.NUM_TRAIN_FACTS

    def test_val_covers_all_seed_edges(self, data_module: SeGALDataModule) -> None:
        total = sum(b.edge_label.shape[0] for b in data_module.val_dataloader())
        assert total == self.NUM_VAL_FACTS

    # ── timestamp-grouped batching ────────────────────────────────────────

    def test_train_batches_have_homogeneous_target_timestamps(
        self, data_module: SeGALDataModule, train_facts: Tensor
    ) -> None:
        """With timestamp grouping (the default), every batch should contain
        seed edges that share the same target timestamp."""
        target_ts = train_facts[:, 3].float()
        for batch in data_module.train_dataloader():
            batch_ts = target_ts[batch.input_id]
            assert (batch_ts == batch_ts[0]).all(), (
                f"Batch contains mixed target timestamps: {batch_ts.unique().tolist()}"
            )

    def test_val_batches_have_homogeneous_target_timestamps(
        self, data_module: SeGALDataModule, val_facts: Tensor
    ) -> None:
        target_ts = val_facts[:, 3].float()
        for batch in data_module.val_dataloader():
            batch_ts = target_ts[batch.input_id]
            assert (batch_ts == batch_ts[0]).all()

    # ── default num_neighbors ─────────────────────────────────────────────

    def test_default_num_neighbors(self, train_facts: Tensor, val_facts: Tensor) -> None:
        dm = SeGALDataModule(train_facts=train_facts, val_facts=val_facts, num_nodes=self.NUM_NODES)
        assert dm.num_neighbors == [128, 128]
        assert dm.batch_size == 32

    # ── precomputed relation labels via DataModule ─────────────────────────

    @pytest.fixture()
    def data_module_with_relation_labels(
        self, train_facts: Tensor, val_facts: Tensor
    ) -> SeGALDataModule:
        all_facts = torch.cat([train_facts, val_facts], dim=0)
        train_history = build_pair_relation_history(train_facts)
        train_relation_labels = build_fact_relation_labels(
            target_facts=train_facts,
            history=train_history,
            num_relations=self.NUM_RELATIONS,
        )
        val_history = build_pair_relation_history(all_facts)
        val_relation_labels = build_fact_relation_labels(
            target_facts=val_facts,
            history=val_history,
            num_relations=self.NUM_RELATIONS,
        )
        return SeGALDataModule(
            train_facts=train_facts,
            val_facts=val_facts,
            num_nodes=self.NUM_NODES,
            train_relation_labels=train_relation_labels,
            val_relation_labels=val_relation_labels,
            num_neighbors=self.NUM_NEIGHBORS,
            batch_size=self.BATCH_SIZE,
        )

    def test_train_batch_has_relation_labels(
        self, data_module_with_relation_labels: SeGALDataModule
    ) -> None:
        batch = next(iter(data_module_with_relation_labels.train_dataloader()))
        assert batch.relation_labels is not None
        assert batch.relation_labels.shape == (batch.edge_label.shape[0], self.NUM_RELATIONS)

    def test_val_batch_has_relation_labels(
        self, data_module_with_relation_labels: SeGALDataModule
    ) -> None:
        batch = next(iter(data_module_with_relation_labels.val_dataloader()))
        assert batch.relation_labels is not None
        assert batch.relation_labels.shape[1] == self.NUM_RELATIONS

    def test_relation_labels_are_binary(
        self, data_module_with_relation_labels: SeGALDataModule
    ) -> None:
        batch = next(iter(data_module_with_relation_labels.train_dataloader()))
        unique_vals = batch.relation_labels.unique()
        assert all(v in (0.0, 1.0) for v in unique_vals.tolist())

    def test_batch_without_relation_labels_has_none(self, data_module: SeGALDataModule) -> None:
        batch = next(iter(data_module.train_dataloader()))
        assert getattr(batch, "relation_labels", None) is None


# ═══════════════════════════════════════════════════════════════════════════════
#  Integration: SeGALDataModule → LitSeGAL
# ═══════════════════════════════════════════════════════════════════════════════


class TestDataModuleLitSeGALIntegration:
    """Verify that real DataModule batches flow through LitSeGAL on GPU."""

    __test__ = True

    NUM_NODES = 50
    NUM_RELATIONS = 5
    EMBEDDING_DIM = 32
    ENCODER_DIM = 32

    @pytest.fixture()
    def data_module(self) -> SeGALDataModule:
        train_facts = _make_synthetic_facts(200, self.NUM_NODES, self.NUM_RELATIONS, 10, seed=10)
        val_facts = _make_synthetic_facts(60, self.NUM_NODES, self.NUM_RELATIONS, 10, seed=20)
        return SeGALDataModule(
            train_facts=train_facts,
            val_facts=val_facts,
            num_nodes=self.NUM_NODES,
            batch_size=8,
            num_neighbors=[5, 3],
            num_negatives=4,
        )

    @pytest.fixture()
    def lit_model(self) -> LitSeGAL:
        cfg = _make_segal_config(
            encoder_dim=self.ENCODER_DIM,
            embedding_dim=self.EMBEDDING_DIM,
            temporal_dim=16,
            scoring_dropout=0.0,
        )
        segal = SeGAL(cfg).to(DEVICE)
        node_embs = torch.randn(self.NUM_NODES, self.EMBEDDING_DIM, device=DEVICE)
        rel_embs = torch.randn(self.NUM_RELATIONS, self.EMBEDDING_DIM, device=DEVICE)
        return LitSeGAL(
            segal=segal,
            node_embeddings=node_embs,
            relation_embeddings=rel_embs,
            loss_fn=RankingRelationLoss(
                entity_loss="MarginRankingLoss",
                entity_loss_kwargs={"margin": 1.0},
                neg_strategy="hardest",
            ),
        ).to(DEVICE)

    def test_training_step_with_real_batch(
        self, lit_model: LitSeGAL, data_module: SeGALDataModule
    ) -> None:
        batch = next(iter(data_module.train_dataloader())).to(DEVICE)
        loss = lit_model.training_step(batch, 0)
        assert loss.dim() == 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_validation_step_with_real_batch(
        self, lit_model: LitSeGAL, data_module: SeGALDataModule
    ) -> None:
        batch = next(iter(data_module.val_dataloader())).to(DEVICE)
        loss = lit_model.validation_step(batch, 0)
        assert loss.dim() == 0
        assert not torch.isnan(loss)

    def test_gradient_flow_with_real_batch(
        self, lit_model: LitSeGAL, data_module: SeGALDataModule
    ) -> None:
        batch = next(iter(data_module.train_dataloader())).to(DEVICE)
        loss = lit_model.training_step(batch, 0)
        loss.backward()
        has_grad = any(p.grad is not None for p in lit_model.segal.parameters() if p.requires_grad)
        assert has_grad

    def test_multiple_train_batches(
        self, lit_model: LitSeGAL, data_module: SeGALDataModule
    ) -> None:
        """Simulate a few training iterations to verify no state corruption."""
        for i, b in enumerate(data_module.train_dataloader()):
            if i >= 3:
                break
            batch = b.to(DEVICE)
            loss = lit_model.training_step(batch, i)
            loss.backward()
            assert not torch.isnan(loss)
            lit_model.zero_grad()

    # ── integration with relation labels ─────────────────────────────────

    @pytest.fixture()
    def data_module_with_relation_labels(self) -> SeGALDataModule:
        train_facts = _make_synthetic_facts(200, self.NUM_NODES, self.NUM_RELATIONS, 10, seed=10)
        val_facts = _make_synthetic_facts(60, self.NUM_NODES, self.NUM_RELATIONS, 10, seed=20)
        all_facts = torch.cat([train_facts, val_facts], dim=0)
        train_history = build_pair_relation_history(train_facts)
        train_relation_labels = build_fact_relation_labels(
            target_facts=train_facts,
            history=train_history,
            num_relations=self.NUM_RELATIONS,
        )
        val_history = build_pair_relation_history(all_facts)
        val_relation_labels = build_fact_relation_labels(
            target_facts=val_facts,
            history=val_history,
            num_relations=self.NUM_RELATIONS,
        )
        return SeGALDataModule(
            train_facts=train_facts,
            val_facts=val_facts,
            num_nodes=self.NUM_NODES,
            train_relation_labels=train_relation_labels,
            val_relation_labels=val_relation_labels,
            batch_size=8,
            num_neighbors=[5, 3],
            num_negatives=4,
        )

    def test_training_step_with_relation_labels_real_batch(
        self, lit_model: LitSeGAL, data_module_with_relation_labels: SeGALDataModule
    ) -> None:
        batch = next(iter(data_module_with_relation_labels.train_dataloader())).to(DEVICE)
        loss = lit_model.training_step(batch, 0)
        assert loss.dim() == 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_validation_step_with_relation_labels_real_batch(
        self, lit_model: LitSeGAL, data_module_with_relation_labels: SeGALDataModule
    ) -> None:
        batch = next(iter(data_module_with_relation_labels.val_dataloader())).to(DEVICE)
        loss = lit_model.validation_step(batch, 0)
        assert loss.dim() == 0
        assert not torch.isnan(loss)
