from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass
class SupervisedDataset:
    entity_pairs: Tensor  # shape: (N, 2)
    relations: Tensor  # shape: (N,) or (N, R)

    def get_metadata(self) -> dict[str, Any]:
        entity_pairs = self.entity_pairs
        relations = self.relations

        relations_count = relations.sum(0)

        return {
            "num_entity_pairs": int(entity_pairs.shape[0]),
            "entity_pairs_shape": tuple(entity_pairs.shape),
            "relations_shape": tuple(relations.shape),
            # relation stats
            "num_relations_present": int((relations_count > 0).sum().item()),
            "relation_count": tuple(relations_count.tolist()),
            # uniqueness stats
            "num_unique_entities": int(torch.unique(entity_pairs).numel()),
            "num_unique_heads": int(torch.unique(entity_pairs[:, 0]).numel()),
            "num_unique_tails": int(torch.unique(entity_pairs[:, 1]).numel()),
            # sanity
            "is_empty": entity_pairs.shape[0] == 0,
        }
