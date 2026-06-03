from __future__ import annotations

from typing import cast

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor
from torch_geometric.data import Data

from .kg_data import KGData


class KGSubgraph(BaseModel):
    """
    Represents a subgraph extracted from a knowledge graph.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=False, extra="allow"
    )

    node_ids: Tensor = Field(
        ...,
        description="Global node IDs in the subgraph. Shape: [num_nodes_in_subgraph]. "
        "Maps local node indices (0, 1, 2, ...) to global node IDs in the full graph."
        "Each entry corresponds to a node in the subgraph,"
        "enabling mapping back to the original graph.",
    )

    edge_index: Tensor = Field(
        ...,
        description="Edge indices defining the connectivity of the sampled subgraph "
        "Shape: [2, num_total_edges]. "
        "The first row contains source node indices (local to the subgraph), "
        "and the second row contains target node indices (local to the subgraph).",
    )

    edge_type: Tensor = Field(
        ...,
        description="Edge types for all edges in the subgraph. Shape: [num_total_edges]. "
        "Each entry specifies the type or relation of the corresponding edge in `edge_index`. "
        "Edge types are usually integer-encoded, mapping to relations in the knowledge graph "
        "(e.g., 'is_a'). Used to differentiate edge semantics in multi-relational graphs.",
    )

    @classmethod
    def from_kg_data(cls, kg_data: KGData) -> KGSubgraph:
        # Get all unique node IDs that appear in the edges
        all_node_ids = torch.unique(kg_data.edge_index.flatten())

        # Create mapping from global node IDs to local indices
        max_node_id = all_node_ids.max().item()
        global_to_local = torch.full((max_node_id + 1,), -1, dtype=torch.long)
        global_to_local[all_node_ids] = torch.arange(len(all_node_ids))

        # Remap edge_index from global IDs to local indices
        local_edge_index = global_to_local[kg_data.edge_index]

        return cast(
            "KGSubgraph",
            Data(
                node_ids=all_node_ids,  # Maps local indices to global node IDs
                edge_index=local_edge_index,  # Uses local indices [0, 1, 2, ...]
                edge_type=kg_data.edge_type,  # Edge types remain the same
            ),
        )

    @classmethod
    def to_kg_data(
        cls,
        subgraph: KGSubgraph,
        num_nodes: int | None = None,
        num_relations: int | None = None,
    ) -> KGData:
        """
        Convert the subgraph back to KGData format.

        This maps the local node indices back to their original global node IDs
        and creates a KGData object with the original sparse indexing.

        Returns:
            KGData: Knowledge graph data with original global node indexing.
        """
        # Map local edge indices back to global node IDs
        global_edge_index = subgraph.node_ids[subgraph.edge_index]

        if num_nodes is None:
            num_nodes = int(subgraph.node_ids.max().item()) + 1

        if num_relations is None:
            num_relations = int(subgraph.edge_type.max().item()) + 1

        return cast(
            "KGData",
            Data(
                edge_index=global_edge_index,
                edge_type=subgraph.edge_type,
                num_nodes=num_nodes,
                num_relations=num_relations,
            ),
        )
