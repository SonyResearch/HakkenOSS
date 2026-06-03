from __future__ import annotations

from typing import Self, cast

import torch
from pydantic import ConfigDict, Field
from torch import Tensor
from torch_geometric.data import Data

from .kg_subgraph import KGSubgraph


class KGPredictionSubgraph(KGSubgraph):
    """
    Type definition for PyTorch Geometric batch objects used in link prediction tasks.
    Represents a mini-batch of graph data with labeled edges for prediction.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=False, extra="allow"
    )

    edge_label_index: Tensor = Field(
        ...,
        description="Indices of edges to be predicted/evaluated. Shape: [2, num_labeled_edges]. "
        "First row contains source node indices (local), second row contains target indices "
        "(local). These are the edges for which we want to predict existence or compute scores.",
    )

    edge_label: Tensor = Field(
        ...,
        description="Labels/types for the edges in edge_label_index. Shape: [num_labeled_edges]. "
        "For knowledge graphs: relation type IDs (0, 1, 2, ..., num_relations-1). "
        "For link prediction: often binary (0=no_edge, 1=edge) or relation types.",
    )

    def to(self, _device: str | torch.device) -> Self:
        return self

    @classmethod
    def from_facts(
        cls,
        target_facts: Tensor,
        context_facts: Tensor,
        device: str | torch.device | None = None,
    ) -> KGPredictionSubgraph:
        """
        Create a KGPredictionSubgraph from target facts and contextual facts.

        Args:
            target_facts: Tensor of shape [N, 3] containing [subject, relation, object]
                triples that are the target for prediction/evaluation. These will NOT
                be included in the graph structure (edge_index).
            context_facts: Tensor of shape [M, 3] containing [subject, relation, object]
                triples that provide context for the prediction task. These form the
                actual graph structure.

        Returns:
            KGPredictionSubgraph: A subgraph with context_facts as the graph structure
            and target_facts as the edges to be predicted via edge_label_index and edge_label.
        """
        if device is None:
            device = target_facts.device
        all_facts = torch.cat([target_facts, context_facts], dim=0)
        all_node_ids = torch.unique(all_facts[:, [0, 2]].flatten())
        max_node_id = all_node_ids.max().item()

        global_to_local = torch.full((max_node_id + 1,), -1, dtype=torch.long, device=device)
        global_to_local[all_node_ids] = torch.arange(len(all_node_ids), device=device)

        # Remap the context_facts edge_index to use the expanded node set
        context_edge_pairs = context_facts[:, [0, 2]]  # [M, 2]
        local_context_edges = global_to_local[context_edge_pairs]  # [M, 2]
        edge_index = local_context_edges.t().contiguous()  # [2, M]

        # Create edge_label_index for target facts using the same node mapping
        target_edge_pairs = target_facts[:, [0, 2]]  # [N, 2]
        local_target_edges = global_to_local[target_edge_pairs]  # [N, 2]
        edge_label_index = local_target_edges.t().contiguous()  # [2, N]

        # Edge labels are the relation types from target facts
        edge_label = target_facts[:, 1]  # [N] - relation IDs

        # Edge types are from context_facts only
        edge_type = context_facts[:, 1]  # [M] - relation IDs

        return cast(
            "KGPredictionSubgraph",
            Data(
                node_ids=all_node_ids,
                edge_index=edge_index,
                edge_type=edge_type,
                edge_label_index=edge_label_index,
                edge_label=edge_label,
            ).to(device),
        )

    @classmethod
    def from_data(
        cls, data: Data, device: str | torch.device | None = None
    ) -> KGPredictionSubgraph:
        if not hasattr(data, "n_id"):
            raise AttributeError(name="n_id")

        data["node_ids"] = data["n_id"]
        del data["n_id"]

        if device is not None:
            data = data.to(device)

        return cast("KGPredictionSubgraph", data)
