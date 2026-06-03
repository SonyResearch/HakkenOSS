from __future__ import annotations

from typing import cast

import torch
from datasets import DataRepositoryI
from hydra.utils import get_class
from torch import nn
from torch_geometric.nn.models.basic_gnn import BasicGNN

from kge.common.entities import KGPredictionSubgraph
from kge.scores.base import ScoreFn

from .config import GNNKGEConfig


class GNNKGE(nn.Module):
    """A GraphSAGE-based model for knowledge graph embedding."""

    def __init__(
        self,
        embedding_dim: int,
        num_entities: int,
        num_relations: int,
        gnn: BasicGNN,
        score_fn: ScoreFn,
    ):
        super().__init__()
        self.node_emb = nn.Embedding(num_entities, embedding_dim)
        self.rel_emb = nn.Embedding(num_relations, embedding_dim)

        self.gnn = gnn
        self.score_fn = score_fn

    def encode_subgraph(
        self,
        node_ids: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode a subgraph using GraphSAGE convolutions.

        Args:
            node_ids (torch.Tensor): Global node IDs in the subgraph, shape [num_nodes_in_subgraph].
            edge_index (torch.Tensor): Local edge indices, shape [2, num_edges_subgraph].
            edge_type (torch.Tensor | None, optional): Edge type IDs for edges in edge_index,
                shape [num_edges_subgraph]. If None, edge attributes are not used. Defaults to None.

        Returns:
            torch.Tensor: Encoded node embeddings, shape [num_nodes_in_subgraph, hidden].
        """

        edge_attr: torch.Tensor | None = None
        if edge_type is not None:
            edge_attr = self.rel_emb(edge_type)

        x0 = self.node_emb(node_ids)

        return cast(
            "torch.Tensor",
            self.gnn.forward(x=x0, edge_index=edge_index, edge_attr=edge_attr),
        )

    def score(self, pred_subgraph: KGPredictionSubgraph) -> torch.Tensor:
        """Compute scores for a batch of knowledge graph triples based on a subgraph.

        Args:
            facts (torch.Tensor): Triples to score, shape [batch_size, 3], where each row
                contains (subject_id, relation_id, object_id).
            subgraph (KGSubgraph): Subgraph containing node IDs, edge indices, and edge types.

        Returns:
            torch.Tensor: Scores for each triple, shape [batch_size].
        """
        z = self.encode_subgraph(
            node_ids=pred_subgraph.node_ids,
            edge_index=pred_subgraph.edge_index,
            edge_type=pred_subgraph.edge_type,
        )

        target_edge_index = pred_subgraph.edge_label_index
        target_edge_type = pred_subgraph.edge_label

        return self.score_from_z(
            z=z,
            subject_ids=target_edge_index[0],
            relation_ids=target_edge_type,
            object_ids=target_edge_index[1],
        )

    def score_from_z(
        self,
        z: torch.Tensor,
        subject_ids: torch.Tensor,
        relation_ids: torch.Tensor,
        object_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute scores for triples using precomputed node embeddings.

        Args:
            z (torch.Tensor): Encoded node embeddings, shape [num_nodes_in_subgraph, hidden].
            subject_ids (torch.Tensor): Subject node IDs, shape [batch_size].
            relation_ids (torch.Tensor): Relation type IDs, shape [batch_size].
            object_ids (torch.Tensor): Object node IDs, shape [batch_size].

        Returns:
            torch.Tensor: Scores for each triple, shape [batch_size, 1], computed as the sum of
                element-wise multiplication of subject, object, and relation embeddings.
        """
        s_embs = z[subject_ids]  # [B, H]
        o_embs = z[object_ids]  # [B, H]
        r_embs = self.rel_emb(relation_ids)  # [B, H]

        return self.score_fn.all(s_embs, r_embs, o_embs)

    @classmethod
    def from_config(cls, config: GNNKGEConfig, dataset: DataRepositoryI) -> GNNKGE:
        gnn_class: type[BasicGNN] = get_class(config.gnn_class)

        gnn = gnn_class(**config.gnn_kwargs)

        score_fn_class: type[ScoreFn] = get_class(config.score_fn_class)

        if config.score_fn_kwargs is not None:
            score_fn = score_fn_class(**config.score_fn_kwargs)
        else:
            score_fn = score_fn_class()

        return cls(
            embedding_dim=config.embedding_dim,
            num_entities=dataset.num_entities,
            num_relations=dataset.num_relations,
            gnn=gnn,
            score_fn=score_fn,
        )
