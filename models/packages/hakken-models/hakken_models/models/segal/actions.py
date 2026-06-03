"""SeGAL inference actions for scoring facts."""

import torch
from torch import Tensor

from hakken_models.core.configs import SeGALInferenceConfig
from hakken_models.core.constants import DEFAULT_NEGATIVE_SCORE
from hakken_models.core.entities.relation_prediction import RelationPrediction
from hakken_models.data_loaders import TemporalKGLinkNeighborLoader
from hakken_models.datasets.deployment import DatasetDeployment
from hakken_models.models.segal import SeGAL
from hakken_models.models.segal.inference import SeGALInferenceWrapper
from hakken_models.models.segal.text_scoring import segal_score_text


@torch.no_grad()
def segal_score(
    facts: Tensor,
    segal: SeGAL,
    dataset: DatasetDeployment,
    config: SeGALInferenceConfig | None = None,
) -> RelationPrediction:
    """Score facts (s, r, o triples) using SeGAL with temporal context.

    Args:
        facts: Tensor of shape [B, 3] with columns (subject_idx, relation_idx, object_idx).
            Use -1 for missing entities/relations when on_missing=ZERO.
        segal: The SeGAL model.
        dataset: Dataset deployment providing KG context and embeddings.
        config: Inference config (split_names, num_neighbors, return_probs). Defaults
            to SeGALInferenceConfig().

    Returns:
        RelationPrediction with logits (raw scores) and probs (sigmoid-normalized)
        for each fact.
    """
    if config is None:
        config = SeGALInferenceConfig()
    segal = segal.eval()

    device = facts.device

    subjects = facts[:, 0]
    relations = facts[:, 1]
    objects = facts[:, 2]

    entity_pairs = torch.stack([subjects, objects], dim=1)  # shape [B, 2]
    valid_mask = (entity_pairs != -1).all(dim=1)
    all_valid = valid_mask.all().item()

    if all_valid:
        edge_label_index = entity_pairs.t().contiguous()
        edge_label = relations
    else:
        valid_entity_pairs = entity_pairs[valid_mask]
        valid_relations = relations[valid_mask]
        edge_label_index = valid_entity_pairs.t().contiguous()
        edge_label = valid_relations

    batch_size = edge_label_index.shape[1]

    if batch_size == 0:
        logits = torch.full(
            (len(facts),),
            DEFAULT_NEGATIVE_SCORE,
            dtype=torch.float,
            device=device,
        )
        probs = torch.sigmoid(logits).tolist() if config.return_probs else None
        return RelationPrediction.model_construct(logits=logits.tolist(), probs=probs)

    kg_data = dataset.get_kg_data(split_names=config.split_names)
    max_timestamp = kg_data.edge_attr[:, 1].float().max().item()
    target_timestamps = torch.full(
        (batch_size,),
        max_timestamp,
        device=device,
        dtype=torch.float32,
    )

    loader = TemporalKGLinkNeighborLoader(
        data=kg_data,
        num_neighbors=config.num_neighbors,
        edge_label_index=edge_label_index,
        edge_label=edge_label,
        target_timestamps=target_timestamps,
        num_negatives=1,
        batch_size=batch_size,
        shuffle=False,
        group_by_timestamp=False,
    )

    batch = next(iter(loader))
    batch = batch.to(device)

    wrapper = SeGALInferenceWrapper(
        segal=segal,
        node_embeddings=dataset.get_node_embedding_matrix(device=device),
        relation_embeddings=dataset.get_relation_embedding_matrix(device=device),
    )
    wrapper = wrapper.to(device).eval()

    score_output = wrapper.score_batch(batch)
    pos_scores = score_output.pos_scores

    if all_valid:
        logits = pos_scores
    else:
        logits = torch.full(
            (len(facts),),
            DEFAULT_NEGATIVE_SCORE,
            dtype=torch.float,
            device=device,
        )
        logits[valid_mask] = pos_scores

    probs = torch.sigmoid(logits).tolist() if config.return_probs else None

    return RelationPrediction.model_construct(logits=logits.tolist(), probs=probs)


__all__ = ["segal_score", "segal_score_text"]
