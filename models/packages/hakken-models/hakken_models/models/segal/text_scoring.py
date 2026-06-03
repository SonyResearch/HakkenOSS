"""Text-based fact scoring: encode entities/relations via embedder and score."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from langchain_core.embeddings import Embeddings

from hakken_models.core.entities.kg_data import KGData
from hakken_models.core.entities.relation_prediction import RelationPrediction
from hakken_models.models.segal.base import SeGAL
from hakken_models.models.segal.text_schemas import (
    TripleText,
    parse_triple,
)

if TYPE_CHECKING:
    from hakken_models.core.configs import SeGALInferenceConfig


def _build_kg_from_facts(
    facts: list[tuple[int, int, int, float]],
    num_nodes: int,
    device: torch.device,
) -> KGData:
    """Build KGData from indexed facts ``(s_idx, r_idx, o_idx, timestamp)``."""
    if not facts:
        return KGData(
            x=torch.zeros((num_nodes, 1), device=device),
            edge_index=torch.zeros((2, 0), dtype=torch.long, device=device),
            edge_attr=torch.zeros((0, 2), device=device),
            num_nodes=num_nodes,
            n_id=torch.arange(num_nodes, device=device),
        )
    rows = torch.tensor(facts, dtype=torch.float32, device=device)
    edge_index = rows[:, [0, 2]].long().t().contiguous()
    edge_attr = torch.stack([rows[:, 1], rows[:, 3]], dim=1)
    return KGData(
        x=torch.zeros((num_nodes, 1), device=device),
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=num_nodes,
        n_id=torch.arange(num_nodes, device=device),
    )


TripleTextInput = TripleText | list[Any]


@torch.no_grad()
def segal_score_text(
    target_facts: list[TripleTextInput],
    segal: SeGAL,
    embedder: Embeddings,
    context_facts: list[TripleTextInput] | None = None,
    config: SeGALInferenceConfig | None = None,
) -> RelationPrediction:
    """Score facts given as text. All entities/relations are encoded via embedder.

    Context is built from context_facts only.

    Args:
        target_facts: Facts to score. Each is TripleText or [s, r, o] / [s, r, o, t].
        segal: The SeGAL model.
        embedder: Langchain Embeddings for encoding entities and relations.
        context_facts: Optional context facts with timestamps.
        config: Inference config (encode_batch_size, return_probs).

    Returns:
        RelationPrediction with logits and probs for each target fact.
    """
    from hakken_models.core.configs import SeGALInferenceConfig

    if config is None:
        config = SeGALInferenceConfig()
    segal = segal.eval()

    device = next(segal.parameters()).device

    if not target_facts:
        return RelationPrediction.model_construct(logits=[], probs=[])

    parsed_targets = [parse_triple(f) for f in target_facts]
    parsed_context = [parse_triple(f) for f in context_facts] if context_facts else []

    # Pre-compute content strings once per parsed fact.
    target_contents = [
        (s.to_content(), r.to_content(), o.to_content()) for s, r, o, _ in parsed_targets
    ]
    context_contents = [
        (s.to_content(), r.to_content(), o.to_content(), t) for s, r, o, t in parsed_context
    ]

    entity_strings: list[str] = []
    relation_strings: list[str] = []
    for sc, _, oc in target_contents:
        entity_strings.extend([sc, oc])
    for sc, _, oc, _ in context_contents:
        entity_strings.extend([sc, oc])
    for _, rc, _ in target_contents:
        relation_strings.append(rc)
    for _, rc, _, _ in context_contents:
        relation_strings.append(rc)

    # Single embedder call: entities first, then relations.
    unique_entities = list(dict.fromkeys(entity_strings))
    unique_relations = list(dict.fromkeys(relation_strings))
    all_strings = unique_entities + unique_relations

    all_vectors: list[list[float]] = []
    for i in range(0, len(all_strings), config.encode_batch_size):
        all_vectors.extend(embedder.embed_documents(all_strings[i : i + config.encode_batch_size]))

    entity_to_idx = {c: i for i, c in enumerate(unique_entities)}
    n_entities = len(unique_entities)
    rel_to_idx = {c: i for i, c in enumerate(unique_relations)}

    all_matrix = np.array(all_vectors, dtype=np.float32)
    node_emb = torch.tensor(all_matrix[:n_entities], dtype=torch.float32, device=device)
    rel_emb = torch.tensor(all_matrix[n_entities:], dtype=torch.float32, device=device)
    num_nodes = n_entities

    # Build context KG from context_facts.
    context_facts_indices: list[tuple[int, int, int, float]] = [
        (entity_to_idx[sc], rel_to_idx[rc], entity_to_idx[oc], t if t is not None else 0.0)
        for sc, rc, oc, t in context_contents
    ]

    context_kg = _build_kg_from_facts(context_facts_indices, num_nodes, device)
    context_kg.x = node_emb

    # GNN pass (once).
    x_enriched = segal.encode_context(context_kg, rel_emb)

    # Gather target indices and batch-score.
    n_targets = len(parsed_targets)
    s_indices = torch.tensor(
        [entity_to_idx[sc] for sc, _, _ in target_contents], dtype=torch.long, device=device
    )
    r_indices = torch.tensor(
        [rel_to_idx[rc] for _, rc, _ in target_contents], dtype=torch.long, device=device
    )
    o_indices = torch.tensor(
        [entity_to_idx[oc] for _, _, oc in target_contents], dtype=torch.long, device=device
    )

    s_embs = x_enriched[s_indices]
    r_embs = segal.input_proj(rel_emb[r_indices])
    o_embs = x_enriched[o_indices]
    logits = segal.score_embeddings(s_embs, r_embs, o_embs)

    probs = torch.sigmoid(logits).tolist() if config.return_probs else None
    return RelationPrediction.model_construct(logits=logits.tolist(), probs=probs)
