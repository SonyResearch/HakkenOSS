import torch
from torch import Tensor

from hakken_models.core.configs import THiGERInferenceConfig
from hakken_models.core.constants import DEFAULT_NEGATIVE_SCORE
from hakken_models.core.entities.kg_data import KGData
from hakken_models.core.entities.relation_prediction import RelationPrediction
from hakken_models.data_loaders.kg_link_neighbor_loader import KGLinkNeighborLoader
from hakken_models.datasets.deployment import DatasetDeployment
from hakken_models.models.thiger import THiGER


@torch.no_grad()
def thiger_predict(
    entity_pairs: Tensor,
    thiger: THiGER,
    dataset: DatasetDeployment,
    config: THiGERInferenceConfig | None = None,
    logits_cols: list[int] | None = None,
) -> RelationPrediction:
    if config is None:
        config = THiGERInferenceConfig()
    thiger = thiger.eval()

    device = entity_pairs.device

    thiger.to(device)

    kg_data = dataset.get_kg_data(split_names=config.split_names)

    valid_entity_pairs_mask = entity_pairs != -1
    all_valid = torch.all(valid_entity_pairs_mask)

    if all_valid:
        edge_label_index = entity_pairs.t().contiguous()
    else:
        valid_mask = valid_entity_pairs_mask.all(dim=1)
        valid_mask = valid_mask.to(device)
        valid_entity_pairs = entity_pairs[valid_mask]
        edge_label_index = valid_entity_pairs.t().contiguous()

    batch_size = edge_label_index.shape[1]

    loader = KGLinkNeighborLoader(
        data=kg_data,
        num_neighbors=config.num_neighbors,
        batch_size=batch_size,
        edge_label_index=edge_label_index,
        edge_label=torch.zeros([batch_size, 1]),
        shuffle=False,
    )

    data: KGData = next(iter(loader))

    thiger.set_context_temporal_kg(data.to(device))

    if all_valid:
        logits = thiger.compute_logits(entity_pairs)
    else:
        logits = torch.full(
            (entity_pairs.shape[0], thiger.num_relations),
            DEFAULT_NEGATIVE_SCORE,
            dtype=torch.float,
            device=device,
        )
        logits[valid_mask, :] = thiger.compute_logits(valid_entity_pairs)

    if logits_cols is not None:
        logits = logits[:, logits_cols]

    probs = torch.sigmoid(logits).tolist() if config.return_probs else None

    return RelationPrediction.model_construct(logits=logits.tolist(), probs=probs)


@torch.no_grad()
def thiger_score(
    facts: Tensor,
    thiger: THiGER,
    dataset: DatasetDeployment,
    config: THiGERInferenceConfig | None = None,
) -> RelationPrediction:
    if config is None:
        config = THiGERInferenceConfig()
    thiger = thiger.eval()

    device = facts.device

    subjects = facts[:, 0]
    relations = facts[:, 1]
    objects = facts[:, 2]

    thiger.to(device)

    kg_data = dataset.get_kg_data(split_names=config.split_names)

    entity_pairs = torch.stack([subjects, objects], dim=1)  # shape [B, 2]
    valid_entity_pairs_mask = entity_pairs != -1
    all_valid = torch.all(valid_entity_pairs_mask)

    if all_valid:
        edge_label_index = entity_pairs.t().contiguous()
    else:
        valid_mask = valid_entity_pairs_mask.all(dim=1)
        valid_mask = valid_mask.to(device)
        valid_entity_pairs = entity_pairs[valid_mask]
        edge_label_index = valid_entity_pairs.t().contiguous()

    batch_size = edge_label_index.shape[1]
    loader = KGLinkNeighborLoader(
        data=kg_data,
        num_neighbors=config.num_neighbors,
        batch_size=batch_size,
        edge_label_index=edge_label_index,
        edge_label=torch.zeros([batch_size, 1]),
        shuffle=False,
    )

    data: KGData = next(iter(loader))

    thiger.set_context_temporal_kg(data.to(device))

    if all_valid:
        logits_all = thiger.compute_logits(entity_pairs)
    else:
        logits_all = torch.full(
            (entity_pairs.shape[0], thiger.num_relations),
            DEFAULT_NEGATIVE_SCORE,
            dtype=torch.float,
            device=device,
        )
        logits_all[valid_mask, :] = thiger.compute_logits(valid_entity_pairs)

    fact_logits = logits_all[torch.arange(len(facts)), relations]
    probs = torch.sigmoid(fact_logits).tolist() if config.return_probs else None

    return RelationPrediction.model_construct(logits=fact_logits.tolist(), probs=probs)
