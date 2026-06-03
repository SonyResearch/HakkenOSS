"""Lightning training wrapper for SeGAL v2 (GNN-based context)."""

from __future__ import annotations

from typing import Any

import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Tensor, nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import LRScheduler

from hakken_models.core.entities.kg_data import KGData
from hakken_models.core.utils.timing import set_current_timings, timed, timed_scope
from hakken_models.datasets.deployment import DatasetDeployment
from hakken_models.evaluators.metric_hub import MetricHub
from hakken_models.losses import loss_fn_registry
from hakken_models.losses.ranking_relation import RankingRelationLoss
from hakken_models.losses.relation_weights import compute_pos_weight_from_relation_labels

from .base import SeGAL, _global_to_local
from .schemas import ScoreStepOutput


class LitSeGAL(LightningModule):
    """Lightning module for SeGAL v2 with GNN-based context encoding.

    Receives :class:`KGData` batches (with pre-sampled negatives in
    ``neg_edge_label_index``) from
    :class:`TemporalKGLinkNeighborLoader`, runs the GNN once per batch
    to produce context-enriched node embeddings, then scores positive
    target triples and the pre-sampled negatives.

    Args:
        segal: The SeGAL scoring model.
        node_embeddings: ``[num_nodes, D_emb]`` pre-computed node embeddings.
            Stored as a buffer and injected into ``KGData.x`` at batch time.
        relation_embeddings: ``[num_relations, D_emb]`` pre-computed relation
            embeddings.
        loss_fn: :class:`~hakken_models.losses.ranking_relation.RankingRelationLoss`.
        optimizer_cls: Optimizer class.
        optimizer_kwargs: Extra kwargs forwarded to the optimizer.
        scheduler_cls: Optional LR scheduler class.
        scheduler_kwargs: Extra kwargs forwarded to the scheduler.
        val_metric_hub: Optional validation metrics; values are logged under
            ``val/<name>`` at epoch end. Use
            :func:`~hakken_models.models.kge.lightning.build_lit_kge_val_metric_hub`
            with :class:`~hakken_models.evaluators.metric_hub.MetricHubConfig`
            (same bindings as LitKGE: ``pos_scores``, ``neg_scores``, optional
            ``rel_logits`` / ``rel_labels``). When ``None``, mean rank is still
            computed and logged from scores in ``validation_step`` (no hub).
        learn_embeddings: If True, entity/relation tables are ``nn.Embedding``
            of shape ``[num_nodes, encoder_dim]`` / ``[num_rel, encoder_dim]``,
            optimized with a separate LR (``embedding_lr_factor`` × base lr).
        embedding_lr_factor: LR multiplier for embedding tables when
            ``learn_embeddings`` is True.
    """

    def __init__(
        self,
        segal: SeGAL,
        node_embeddings: Tensor,
        relation_embeddings: Tensor,
        loss_fn: RankingRelationLoss,
        optimizer_cls: type[Optimizer] = Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
        scheduler_cls: type[LRScheduler] | None = None,
        scheduler_kwargs: dict[str, Any] | None = None,
        val_metric_hub: MetricHub | None = None,
        learn_embeddings: bool = False,
        embedding_lr_factor: float = 0.1,
    ) -> None:
        super().__init__()

        self.segal = segal
        self.loss_fn = loss_fn
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.scheduler_cls = scheduler_cls
        self.scheduler_kwargs = scheduler_kwargs or {}
        self.learn_embeddings = learn_embeddings
        self.embedding_lr_factor = embedding_lr_factor

        self.loss_fn_name = loss_fn.__class__.__name__
        self.optimizer_name = optimizer_cls.__name__

        enc = segal.config.encoder_dim
        if learn_embeddings:
            if node_embeddings.shape[1] != enc or relation_embeddings.shape[1] != enc:
                raise ValueError(
                    "learn_embeddings requires last dim == segal.config.encoder_dim for both tables "
                    f"(got node {node_embeddings.shape[1]}, rel {relation_embeddings.shape[1]}, encoder_dim={enc})."
                )

        num_nodes = node_embeddings.shape[0]
        num_rel = relation_embeddings.shape[0]
        if learn_embeddings:
            self.node_emb = nn.Embedding(num_nodes, enc)
            self.node_emb.weight.data.copy_(node_embeddings)
            self.rel_emb = nn.Embedding(num_rel, enc)
            self.rel_emb.weight.data.copy_(relation_embeddings)
        else:
            self.register_buffer("node_embs", node_embeddings)
            self.register_buffer("rel_embs", relation_embeddings)

        self.val_metric_hub = val_metric_hub

        self.save_hyperparameters(
            ignore=[
                "segal",
                "node_embeddings",
                "relation_embeddings",
                "loss_fn",
                "optimizer_cls",
                "scheduler_cls",
                "val_metric_hub",
            ],
            logger=False,
        )
        self._train_timings: list[dict[str, float]] = []
        self._val_timings: list[dict[str, float]] = []

    def _node_table(self) -> Tensor:
        if self.learn_embeddings:
            return self.node_emb.weight
        return self.node_embs

    def _rel_table(self) -> Tensor:
        if self.learn_embeddings:
            return self.rel_emb.weight
        return self.rel_embs

    def embedding_tables(self) -> tuple[Tensor, Tensor]:
        """Full node and relation embedding matrices (buffers or ``nn.Embedding`` weights)."""
        return self._node_table(), self._rel_table()

    # ── batch unpacking ──────────────────────────────────────────────────

    @timed("prepare_batch_ms")
    def _prepare_batch(self, batch: KGData) -> KGData:
        """Inject node embeddings into the batch."""
        batch.x = self._node_table()[batch.n_id]
        return batch

    def _extract_targets(self, batch: KGData) -> tuple[Tensor, Tensor, Tensor]:
        """Extract target (subject, object, relation) from the batch.

        Returns:
            Tuple of ``(subject_global, object_global, relation_idx)``
            each ``[B]``.
        """
        subject_global = batch.edge_label_index[0]
        object_global = batch.edge_label_index[1]
        relation_idx = batch.edge_label.long()
        return subject_global, object_global, relation_idx

    # ── scoring helpers ──────────────────────────────────────────────────

    def _score_step(
        self,
        batch: KGData,
        *,
        override_x: Tensor | None = None,
    ) -> ScoreStepOutput:
        """Run one forward step: GNN + score positives + score negatives.

        Entity-corrupted negatives are read from
        ``batch.neg_edge_label_index`` (shape ``[2, B, K]``).

        When ``batch.relation_labels`` is present (a ``[B, R]`` multi-hot
        tensor), all relations are scored for each ``(s, o)`` pair,
        producing ``rel_logits`` of shape ``[B, R]`` alongside the
        entity-negative scores.

        Args:
            override_x: If provided, used as ``batch.x`` instead of
                looking up the node embedding table.

        Returns:
            :class:`ScoreStepOutput` containing ``pos_scores [B]``,
            ``neg_scores [B, K]``, and optionally ``rel_logits [B, R]``
            with ``rel_labels [B, R]``.
        """
        if override_x is not None:
            batch.x = override_x
        else:
            batch = self._prepare_batch(batch)

        s_global, o_global, r_idx = self._extract_targets(batch)
        rel_table = self._rel_table()
        r_emb = self.segal.input_proj(rel_table[r_idx])

        with timed_scope("encode_context_ms"):
            x_enriched = self.segal.encode_context(batch, rel_table)

        s_local = _global_to_local(s_global, batch.n_id)
        o_local = _global_to_local(o_global, batch.n_id)

        with timed_scope("scoring_ms"):
            pos_scores = self.segal.score_embeddings(
                x_enriched[s_local], r_emb, x_enriched[o_local]
            )

            # -- entity-corrupted negatives --
            neg_s_global = batch.neg_edge_label_index[0]
            neg_o_global = batch.neg_edge_label_index[1]

            batch_size, num_negatives = neg_s_global.shape
            neg_s_flat = neg_s_global.reshape(-1)
            neg_o_flat = neg_o_global.reshape(-1)

            neg_s_local = _global_to_local(neg_s_flat, batch.n_id)
            neg_o_local = _global_to_local(neg_o_flat, batch.n_id)

            r_emb_expanded = (
                r_emb.unsqueeze(1)
                .expand(batch_size, num_negatives, -1)
                .reshape(batch_size * num_negatives, -1)
            )

            neg_scores_flat = self.segal.score_embeddings(
                x_enriched[neg_s_local], r_emb_expanded, x_enriched[neg_o_local]
            )
            neg_scores = neg_scores_flat.view(batch_size, num_negatives)

        # -- multi-label relation scoring --
        rel_logits: Tensor | None = None
        rel_labels: Tensor | None = getattr(batch, "relation_labels", None)

        if rel_labels is not None:
            with timed_scope("rel_scoring_ms"):
                all_r_emb = self.segal.input_proj(rel_table)
                num_rel = all_r_emb.shape[0]

                s_emb = x_enriched[s_local].unsqueeze(1).expand(batch_size, num_rel, -1)
                o_emb = x_enriched[o_local].unsqueeze(1).expand(batch_size, num_rel, -1)
                r_exp = all_r_emb.unsqueeze(0).expand(batch_size, num_rel, -1)

                rel_logits = self.segal.score_embeddings(
                    s_emb.reshape(batch_size * num_rel, -1),
                    r_exp.reshape(batch_size * num_rel, -1),
                    o_emb.reshape(batch_size * num_rel, -1),
                ).view(batch_size, num_rel)

        return ScoreStepOutput(pos_scores, neg_scores, rel_logits, rel_labels)

    # ── loss ─────────────────────────────────────────────────────────────

    @timed("compute_loss_ms")
    def compute_loss(
        self,
        pos_scores: Tensor,
        neg_scores: Tensor,
        rel_logits: Tensor | None = None,
        rel_labels: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute combined entity-ranking + relation-classification loss.

        Returns:
            Tuple of (total_loss, loss_dict) with keys ``entity`` and optionally ``relation``.
        """
        return self.loss_fn(pos_scores, neg_scores, rel_logits=rel_logits, rel_labels=rel_labels)

    # ── optimizer / scheduler ────────────────────────────────────────────

    def configure_optimizers(self) -> OptimizerLRScheduler:
        if self.learn_embeddings:
            opt_kw = dict(self.optimizer_kwargs)
            base_lr = float(opt_kw.pop("lr", 1e-3))
            emb_lr = base_lr * self.embedding_lr_factor
            param_groups: list[dict[str, Any]] = [
                {"params": self.segal.parameters(), "lr": base_lr, **opt_kw},
                {"params": self.node_emb.parameters(), "lr": emb_lr, **opt_kw},
                {"params": self.rel_emb.parameters(), "lr": emb_lr, **opt_kw},
            ]
            optimizer = self.optimizer_cls(param_groups)
        else:
            optimizer = self.optimizer_cls(params=self.segal.parameters(), **self.optimizer_kwargs)

        if self.scheduler_cls is not None:
            # Lightning-specific keys are passed to the lr_scheduler dict, not to the scheduler
            lightning_keys = {"interval", "frequency", "monitor"}
            scheduler_kw = {
                k: v for k, v in self.scheduler_kwargs.items() if k not in lightning_keys
            }
            scheduler = self.scheduler_cls(optimizer, **scheduler_kw)
            lr_scheduler_config: dict[str, Any] = {
                "scheduler": scheduler,
                "interval": self.scheduler_kwargs.get("interval", "epoch"),
                "frequency": self.scheduler_kwargs.get("frequency", 1),
            }
            if "monitor" in self.scheduler_kwargs:
                lr_scheduler_config["monitor"] = self.scheduler_kwargs["monitor"]
            return {
                "optimizer": optimizer,
                "lr_scheduler": lr_scheduler_config,
            }

        return optimizer

    def on_validation_epoch_start(self) -> None:
        if self.val_metric_hub is not None:
            self.val_metric_hub.to(self.device)

    # ── training / validation ────────────────────────────────────────────

    def _log_val_metric_hub_results(self, results: dict[str, Any]) -> None:
        log_kw = dict(on_step=False, on_epoch=True, sync_dist=True)
        for name, value in results.items():
            if isinstance(value, Tensor):
                v = value.detach()
                scalar = v.item() if v.numel() == 1 else v.float().mean().item()
            else:
                scalar = float(value)
            prog_bar = name == "mean_rank"
            self.log(f"val/{name}", scalar, prog_bar=prog_bar, **log_kw)

    def _log_step_metrics(
        self,
        prefix: str,
        batch: KGData,
        out: ScoreStepOutput,
        loss: Tensor,
        loss_dict: dict[str, Tensor],
        *,
        mean_rank: Tensor | None = None,
        mean_rank_random: Tensor | None = None,
    ) -> None:
        """Log step metrics with the given prefix (e.g. 'train' or 'val').

        Args:
            prefix: Metric name prefix (e.g. 'train', 'val').
            batch: The KGData batch.
            out: Score step output.
            loss: Scalar loss tensor.
            loss_dict: Component losses (entity, rel) for logging.
            mean_rank: Optional mean rank to log (validation only).
            mean_rank_random: Optional random baseline mean rank (validation only).
        """
        batch_size = out.pos_scores.size(0)
        log_kw = dict(on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)

        self.log(f"{prefix}/loss", loss, **log_kw)
        self.log(f"{prefix}/entity_loss", loss_dict["entity"], **log_kw)
        if "relation" in loss_dict:
            self.log(f"{prefix}/rel_loss", loss_dict["relation"], **log_kw)

        num_context_facts = batch.edge_index.size(1)
        avg_num_context_facts = num_context_facts / batch_size
        self.log(f"{prefix}/context_facts", float(num_context_facts), **log_kw)
        self.log(f"{prefix}/avg_context_facts_per_target", float(avg_num_context_facts), **log_kw)
        self.log(f"{prefix}/context_nodes", float(batch.num_nodes), **log_kw)

        num_negatives = out.neg_scores.shape[1]
        self.log(f"{prefix}/num_negatives", num_negatives, **log_kw)

        if batch.edge_attr.size(1) >= 2:
            edge_ts = batch.edge_attr[:, 1].float()
            self.log(f"{prefix}/avg_timestamp", edge_ts.mean().item(), **log_kw)
            self.log(f"{prefix}/num_unique_timestamps", float(edge_ts.unique().numel()), **log_kw)

        target_ts = getattr(batch, "target_timestamps", None)
        if target_ts is not None:
            self.log(
                f"{prefix}/num_unique_target_timestamps",
                float(target_ts.unique().numel()),
                **log_kw,
            )

        if mean_rank is not None:
            self.log(f"{prefix}/mean_rank", mean_rank, prog_bar=True, **log_kw)
        if mean_rank_random is not None:
            self.log(
                f"{prefix}/mean_rank_random",
                mean_rank_random,
                prog_bar=(self.val_metric_hub is None),
                **log_kw,
            )

    def training_step(self, batch: KGData, _batch_idx: int) -> Tensor | None:
        timings: dict[str, float] = {}
        set_current_timings(timings)
        try:
            out = self._score_step(batch)
            loss, loss_dict = self.compute_loss(
                out.pos_scores,
                out.neg_scores,
                out.rel_logits,
                out.rel_labels,
            )
            self._train_timings.append(timings)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            set_current_timings(None)
            if isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return None
            raise
        finally:
            set_current_timings(None)
        self._log_step_metrics("train", batch, out, loss, loss_dict)
        return loss

    @torch.no_grad()
    def _compute_mean_rank(self, pos_scores: Tensor, neg_scores: Tensor) -> Tensor:
        """Approximate mean rank: 1 + #{negatives scoring higher}.

        Returns inf when scores contain NaN/Inf, so early stopping (minimize
        mean_rank) treats broken outputs as worst-case rather than falsely perfect.
        """
        if not torch.isfinite(pos_scores).all() or not torch.isfinite(neg_scores).all():
            return pos_scores.new_tensor(float("inf"))
        num_better = (neg_scores > pos_scores.unsqueeze(1)).sum(dim=1).float()
        return (1.0 + num_better).mean()

    def validation_step(self, batch: KGData, _batch_idx: int) -> Tensor:
        timings: dict[str, float] = {}
        set_current_timings(timings)
        try:
            out = self._score_step(batch)
            loss, loss_dict = self.compute_loss(
                out.pos_scores,
                out.neg_scores,
                out.rel_logits,
                out.rel_labels,
            )
            self._val_timings.append(timings)
        finally:
            set_current_timings(None)

        if self.val_metric_hub is not None:
            hub_kw: dict[str, Any] = {
                "pos_scores": out.pos_scores.detach(),
                "neg_scores": out.neg_scores.detach(),
            }
            if out.rel_logits is not None and out.rel_labels is not None:
                hub_kw["rel_logits"] = out.rel_logits.detach()
                hub_kw["rel_labels"] = out.rel_labels.detach()
            self.val_metric_hub.update(**hub_kw)

        mean_rank: Tensor | None = None
        if self.val_metric_hub is None:
            mean_rank = self._compute_mean_rank(out.pos_scores, out.neg_scores)

        random_x = torch.randn_like(batch.x)
        rand_out = self._score_step(batch, override_x=random_x)
        mean_rank_random = self._compute_mean_rank(rand_out.pos_scores, rand_out.neg_scores)

        self._log_step_metrics(
            "val",
            batch,
            out,
            loss,
            loss_dict,
            mean_rank=mean_rank,
            mean_rank_random=mean_rank_random,
        )
        return loss

    def on_train_epoch_end(self, *args, **kwargs) -> None:
        opts = self.optimizers()
        optimizer = opts[0] if isinstance(opts, list) else opts
        lr = optimizer.param_groups[0]["lr"]
        self.log("learning_rate", lr, on_step=False, on_epoch=True, sync_dist=True)
        if not self._train_timings:
            return
        keys = list(self._train_timings[0].keys())
        n = len(self._train_timings)
        for key in keys:
            mean_ms = sum(t[key] for t in self._train_timings) / n
            self.log(f"timing/train_{key}", mean_ms, on_step=False, on_epoch=True)
        self._train_timings.clear()

    def on_validation_epoch_end(self, *args, **kwargs) -> None:
        if self.val_metric_hub is not None:
            results = self.val_metric_hub.compute_and_reset()
            self._log_val_metric_hub_results(results)

        if not self._val_timings:
            return
        keys = list(self._val_timings[0].keys())
        n = len(self._val_timings)
        for key in keys:
            mean_ms = sum(t[key] for t in self._val_timings) / n
            self.log(f"timing/val_{key}", mean_ms, on_step=False, on_epoch=True)
        self._val_timings.clear()


# ── Factory ──────────────────────────────────────────────────────────────────


def create_lit_segal(
    segal: SeGAL,
    node_embeddings: Tensor,
    relation_embeddings: Tensor,
    loss_config: dict[str, Any],
    optimizer_config: dict[str, Any],
    scheduler_config: dict[str, Any] | None = None,
    neg_strategy: str = "hardest",
    dataset: DatasetDeployment | None = None,
    val_metric_hub: MetricHub | None = None,
    learn_embeddings: bool = False,
    embedding_lr_factor: float = 0.1,
) -> LitSeGAL:
    """Build a :class:`LitSeGAL` from config dicts.

    Args:
        segal: SeGAL scoring model.
        node_embeddings: ``[num_nodes, D_emb]``.
        relation_embeddings: ``[num_relations, D_emb]``.
        loss_config: ``{"name": "RankingRelationLoss", "kwargs": {...}}`` or
            ``{"name": "MarginRankingLoss", "kwargs": {...}}`` (wrapped as entity loss).
        optimizer_config: ``{"name": "Adam", "kwargs": {"lr": 1e-3}}``.
        scheduler_config: Optional scheduler config dict.
        neg_strategy: Used only when ``loss_config["name"]`` is not
            ``RankingRelationLoss`` (plain entity loss wrapped in
            :class:`~hakken_models.losses.ranking_relation.RankingRelationLoss`).
            For ``RankingRelationLoss``, set ``neg_strategy`` in
            ``loss_config["kwargs"]`` instead.
        dataset: Optional dataset for computing pos_weight from train relation
            labels when ``relation_loss_kwargs.pos_weight_from_data`` is True.
        val_metric_hub: Optional validation :class:`MetricHub` (e.g. from
            ``build_lit_kge_val_metric_hub(cfg, num_relations=...)``).
        learn_embeddings: Train entity/relation tables (encoder_dim) with a separate LR.
        embedding_lr_factor: ``embedding_lr = base_lr * embedding_lr_factor`` for tables.

    Returns:
        Configured :class:`LitSeGAL`.
    """
    loss_name = loss_config["name"]
    loss_kwargs = dict(loss_config.get("kwargs", {}))

    if loss_name == "RankingRelationLoss":
        rel_kwargs = dict(loss_kwargs.get("relation_loss_kwargs", {}))
        if rel_kwargs.pop("pos_weight_from_data", False) and dataset is not None:
            if dataset.has_relation_labels:
                labels = dataset.get_relation_labels_tensor("train")
                pos_weight = compute_pos_weight_from_relation_labels(labels)
                rel_kwargs["pos_weight"] = pos_weight
        loss_kwargs["relation_loss_kwargs"] = rel_kwargs
        loss_fn = loss_fn_registry.create(loss_name, **loss_kwargs)
    else:
        loss_fn = RankingRelationLoss(
            entity_loss=loss_name,
            entity_loss_kwargs=loss_kwargs,
            neg_strategy=neg_strategy,
        )

    optimizer_cls = getattr(torch.optim, optimizer_config["name"])

    scheduler_cls = None
    if scheduler_config:
        scheduler_cls = getattr(torch.optim.lr_scheduler, scheduler_config["name"])

    return LitSeGAL(
        segal=segal,
        node_embeddings=node_embeddings,
        relation_embeddings=relation_embeddings,
        loss_fn=loss_fn,
        optimizer_cls=optimizer_cls,
        optimizer_kwargs=optimizer_config.get("kwargs", {}),
        scheduler_cls=scheduler_cls,
        scheduler_kwargs=scheduler_config.get("kwargs", {}) if scheduler_config else None,
        val_metric_hub=val_metric_hub,
        learn_embeddings=learn_embeddings,
        embedding_lr_factor=embedding_lr_factor,
    )
