from __future__ import annotations

from typing import cast

from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor
from torch_geometric.data import Data


class KGData(BaseModel):
    """
    Representation of a Knowledge Graph as a PyTorch Geometric compatible data structure.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    edge_index: Tensor = Field(
        ...,
        description="Graph connectivity in COO format with shape [2, num_edges]. "
        "First row contains source nodes, second row contains target nodes.",
    )
    edge_type: Tensor = Field(
        ...,
        description="Relation type for each edge with shape [num_edges]. "
        "Each value is a non-negative integer corresponding to a relation type ID.",
    )
    num_nodes: int = Field(..., ge=0, description="Number of unique nodes in the graph.")
    num_relations: int = Field(
        ..., ge=1, description="Number of unique relation types in the graph."
    )

    @classmethod
    def from_facts(
        cls,
        facts: Tensor,
        num_nodes: int | None = None,
        num_relations: int | None = None,
    ) -> KGData:
        """
        Build a compact graph from knowledge graph triples.

        Args:
            facts (torch.Tensor): Tensor of shape [N, 3] containing
                [subject, relation, object] triples containing long IDs
            num_nodes (int, optional): Total number of nodes in the graph. If None,
                inferred as max(node_id) + 1 from the facts.
            num_relations (int, optional): Total number of relation types. If None,
                inferred as max(relation_id) + 1 from the facts.

        Returns:
            KGData: Knowledge graph data structure with edge_index containing original
                node indices, edge_type with relation IDs, and metadata about graph size.

        Note:
            The resulting edge_index preserves the original node indexing scheme from
            the input facts without any node ID remapping or compaction to consecutive
            integers. This means node IDs may be sparse (e.g., [0, 5, 100]) rather
            than dense (e.g., [0, 1, 2]).
        """
        edge_pairs: Tensor = facts[:, [0, 2]]  # [N, 2]
        rel_ids: Tensor = facts[:, 1]  # [N]

        edge_index = edge_pairs.t().contiguous()

        if num_nodes is None:
            # Find the maximum node ID to determine number of nodes
            num_nodes = int(edge_pairs.max().item()) + 1

        if num_relations is None:
            num_relations = int(rel_ids.max().item()) + 1

        if num_nodes < edge_index.max() + 1:
            msg = f"{num_nodes} < edge_index.max() + 1"
            raise IndexError(msg)
        data = Data(
            edge_index=edge_index,
            num_relations=num_relations,
            num_nodes=num_nodes,
            edge_type=rel_ids,
        )

        return cast("KGData", data)
