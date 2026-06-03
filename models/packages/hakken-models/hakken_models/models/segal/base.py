"""SeGAL: Semantic Graph-Aware Link scorer.

Predicts a plausibility score for a target fact ``(s, r, o)`` given a
context subgraph of temporally preceding facts.  The context subgraph
is processed by a GNN with temporal edge features; the resulting
context-enriched node embeddings for ``s`` and ``o`` are concatenated
with the relation embedding and scored by an MLP.

Entities and relations are represented as pre-computed dense embeddings
stored in the context graph's node features (``x``) and a relation
buffer, respectively.  Timestamps on context edges are encoded with a
continuous temporal encoder and injected as GNN edge features alongside
relation embeddings.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from hakken_models.core.entities.kg_data import KGData
from hakken_models.models.gnn import gnn_registry

from .config import SeGALConfig
from .schemas import ScoreBatchOutput
from .temporal import TemporalEncoder


def _global_to_local(global_ids: Tensor, n_id: Tensor) -> Tensor:
    """Map global node IDs to local indices within a sampled subgraph."""
    n_id_sorted, sort_perm = n_id.sort()
    positions = torch.searchsorted(n_id_sorted, global_ids)
    return sort_perm[positions]


def _build_scoring_mlp(
    in_dim: int,
    hidden_dim: int,
    num_layers: int = 2,
    dropout: float = 0.1,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    dim = in_dim
    for _ in range(num_layers):
        layers += [nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
        dim = hidden_dim
    layers.append(nn.Linear(hidden_dim, 1))
    return nn.Sequential(*layers)


class SeGAL(nn.Module):
    """Semantic Graph-Aware Link scorer.

    Architecture
    ------------
    1. **Input projection** — maps pre-computed embeddings from
       ``D_emb`` to ``D_enc``.
    2. **Temporal encoder** — encodes continuous edge timestamps into
       dense vectors used as GNN edge features.
    3. **GNN** — message-passing over the context subgraph with
       temporal + relational edge features, producing context-enriched
       node embeddings.
    4. **Scoring MLP** — ``[s_enriched ∥ r_emb ∥ o_enriched] → scalar``.

    Attributes:
        config: The configuration used to build the model.
        temporal_encoder: Continuous temporal encoder for edge timestamps.
        gnn: Graph neural network for context aggregation.
        scoring_mlp: ``[s ∥ r ∥ o] → scalar`` plausibility logit.
    """

    def __init__(self, config: SeGALConfig) -> None:
        super().__init__()
        self.config = config

        embedding_dim = (
            config.encoder_dim if config.learn_embeddings else config.embedder.embedding_dim
        )
        layers: list[nn.Module] = [nn.LayerNorm(embedding_dim)]
        if embedding_dim != config.encoder_dim:
            layers.append(nn.Linear(embedding_dim, config.encoder_dim))
        else:
            layers.append(nn.Identity())
        self.input_proj: nn.Module = nn.Sequential(*layers)

        self.temporal_encoder = TemporalEncoder(
            embedding_dim=config.temporal.embedding_dim,
            learnable_frequencies=config.temporal.learnable_frequencies,
            num_sinusoidal=config.temporal.num_sinusoidal,
        )

        if config.edge_feature_mode == "cat":
            edge_dim = config.encoder_dim + config.temporal.embedding_dim
        elif config.edge_feature_mode == "add":
            if config.encoder_dim != config.temporal.embedding_dim:
                raise ValueError(
                    "encoder_dim must equal temporal embedding_dim when "
                    f"edge_feature_mode='add', got {config.encoder_dim} vs "
                    f"{config.temporal.embedding_dim}"
                )
            edge_dim = config.encoder_dim
        else:
            raise ValueError(f"Unknown edge_feature_mode: {config.edge_feature_mode!r}")

        self.gnn = gnn_registry.create(
            config.gnn.name,
            in_channels=config.encoder_dim,
            out_channels=config.encoder_dim,
            edge_dim=edge_dim,
            **config.gnn.kwargs,
        )

        self.scoring_mlp = _build_scoring_mlp(
            in_dim=3 * config.encoder_dim,
            hidden_dim=config.scoring.hidden_dim,
            num_layers=config.scoring.num_layers,
            dropout=config.scoring.dropout,
        )

        if config.use_inverse_relations:
            self.inv_emb = nn.Parameter(torch.empty(config.encoder_dim))
            # 1-D offset added to relation embeddings on reverse edges; xavier requires dim>=2
            std = (2.0 / max(config.encoder_dim, 1)) ** 0.5
            nn.init.normal_(self.inv_emb, mean=0.0, std=std)
        else:
            self.inv_emb = None

    # ── edge feature construction ─────────────────────────────────────────

    def _build_edge_features(
        self,
        rel_embs: Tensor,
        timestamps: Tensor,
    ) -> Tensor:
        """Combine relation embeddings and temporal embeddings into edge features.

        Args:
            rel_embs: ``[E, D_enc]`` relation embeddings for each context edge.
            timestamps: ``[E]`` float timestamps for each context edge.

        Returns:
            ``[E, edge_dim]`` edge features for the GNN.
        """
        t_emb = self.temporal_encoder(timestamps)

        if self.config.edge_feature_mode == "cat":
            return torch.cat([rel_embs, t_emb], dim=-1)
        return rel_embs + t_emb

    # ── context encoding via GNN ──────────────────────────────────────────

    def encode_context(
        self,
        context_graph: KGData,
        relation_embeddings: Tensor,
    ) -> Tensor:
        """Run GNN on the context subgraph to produce enriched node embeddings.

        When ``use_inverse_relations`` is True and ``edge_attr`` has a third column
        (direction: 0 = forward, 1 = reverse), reverse edges use
        ``r_emb + inv_emb`` for the relation part of the edge features.

        Args:
            context_graph: Subgraph with ``x`` (pre-computed node features),
                ``edge_index``, ``edge_attr`` (``[E, >=2]``: col 0 = relation
                index, col 1 = timestamp; optional col 2 = direction 0/1).
            relation_embeddings: ``[num_relations, D_emb]`` pre-computed relation
                embedding matrix.

        Returns:
            ``[N_sub, D_enc]`` context-enriched node embeddings.
        """
        x_proj = self.input_proj(context_graph.x)

        edge_rel_ids = context_graph.edge_attr[:, 0].long()
        edge_timestamps = context_graph.edge_attr[:, 1].float()

        rel_embs = self.input_proj(relation_embeddings[edge_rel_ids])
        if self.config.use_inverse_relations and self.inv_emb is not None:
            ea = context_graph.edge_attr
            if ea.shape[1] >= 3:
                reverse_mask = ea[:, 2].bool()
            else:
                # No direction column (e.g. legacy 2-col [rel, time]): treat as all forward
                reverse_mask = torch.zeros(ea.shape[0], dtype=torch.bool, device=rel_embs.device)
            rel_embs = rel_embs + (
                self.inv_emb.unsqueeze(0) * reverse_mask.unsqueeze(1).to(rel_embs.dtype)
            )
        edge_features = self._build_edge_features(rel_embs, edge_timestamps)

        return self.gnn(
            x=x_proj,
            edge_index=context_graph.edge_index,
            edge_attr=edge_features,
        )

    # ── scoring ───────────────────────────────────────────────────────────

    def score(
        self,
        subject_local_idx: Tensor,
        relation_emb: Tensor,
        target_local_idx: Tensor,
        context_graph: KGData,
        relation_embeddings: Tensor,
    ) -> Tensor:
        """Score target triples against a temporal context subgraph.

        Args:
            subject_local_idx: ``[B]`` local node indices of subjects in
                *context_graph*.
            relation_emb: ``[B, D_enc]`` pre-computed, projected relation
                embeddings for the target triples.
            target_local_idx: ``[B]`` local node indices of objects/targets
                in *context_graph*.
            context_graph: Temporal context subgraph.  Must satisfy:

                * Does **not** contain the target facts.
                * All edge timestamps ``<= min(t_targets)``.
            relation_embeddings: ``[num_relations, D_emb]`` pre-computed relation
                embedding matrix (used for GNN edge features).

        Returns:
            ``[B]`` plausibility logits.
        """
        x_enriched = self.encode_context(context_graph, relation_embeddings)

        s_emb = x_enriched[subject_local_idx]
        o_emb = x_enriched[target_local_idx]

        x = torch.cat([s_emb, relation_emb, o_emb], dim=-1)
        return self.scoring_mlp(x).squeeze(-1)

    # ── low-level scoring from pre-computed embeddings ──────────────────

    def score_embeddings(
        self,
        s_emb: Tensor,
        r_emb: Tensor,
        o_emb: Tensor,
    ) -> Tensor:
        """Score from already-computed embeddings (no GNN call).

        Useful when the caller has already run :meth:`encode_context`
        and wants to score multiple sets of (s, r, o) against the same
        enriched node embeddings (e.g. positives + negatives).

        Args:
            s_emb: ``[B, D_enc]`` subject embeddings.
            r_emb: ``[B, D_enc]`` relation embeddings.
            o_emb: ``[B, D_enc]`` object embeddings.

        Returns:
            ``[B]`` plausibility logits.
        """
        x = torch.cat([s_emb, r_emb, o_emb], dim=-1)
        return self.scoring_mlp(x).squeeze(-1)

    # ── batch scoring for evaluation ───────────────────────────────────────

    def score_batch(
        self,
        batch: KGData,
        node_embeddings: Tensor,
        relation_embeddings: Tensor,
    ) -> ScoreBatchOutput:
        """Score a batch of positives and negatives for evaluation.

        Args:
            batch: KGData with edge_label_index, neg_edge_label_index.
            node_embeddings: [num_nodes, D_emb] pre-computed node embeddings.
            relation_embeddings: [num_relations, D_emb] pre-computed relation embeddings.

        Returns:
            ScoreBatchOutput with pos_scores [B] and neg_scores [B, K].
        """
        batch.x = node_embeddings[batch.n_id]

        s_global = batch.edge_label_index[0]
        o_global = batch.edge_label_index[1]
        r_idx = batch.edge_label.long()

        r_emb = self.input_proj(relation_embeddings[r_idx])
        x_enriched = self.encode_context(batch, relation_embeddings)

        s_local = _global_to_local(s_global, batch.n_id)
        o_local = _global_to_local(o_global, batch.n_id)

        pos_scores = self.score_embeddings(x_enriched[s_local], r_emb, x_enriched[o_local])

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

        neg_scores_flat = self.score_embeddings(
            x_enriched[neg_s_local], r_emb_expanded, x_enriched[neg_o_local]
        )
        neg_scores = neg_scores_flat.view(batch_size, num_negatives)

        return ScoreBatchOutput(pos_scores, neg_scores)

    # ── forward (training shortcut) ───────────────────────────────────────

    def forward(
        self,
        subject_local_idx: Tensor,
        relation_emb: Tensor,
        target_local_idx: Tensor,
        context_graph: KGData,
        relation_embeddings: Tensor,
    ) -> Tensor:
        """Alias for :meth:`score` — used by Lightning's training loop."""
        return self.score(
            subject_local_idx=subject_local_idx,
            relation_emb=relation_emb,
            target_local_idx=target_local_idx,
            context_graph=context_graph,
            relation_embeddings=relation_embeddings,
        )
