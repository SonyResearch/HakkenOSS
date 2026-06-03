from __future__ import annotations

from typing import Any, cast

import polars as pl
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.typing import OptTensor
from torch_geometric.utils import subgraph


def assert_is_kg_data(obj: object) -> None:
    assert isinstance(obj, Data)
    required_fields = ["x", "edge_index", "edge_attr", "num_nodes"]
    for field in required_fields:
        assert hasattr(obj, field), f"KGData missing required field '{field}'"

    assert obj.edge_index.shape[0] == 2
    assert obj.edge_index.shape[1] == obj.edge_attr.shape[0]


def assert_is_kg_data_with_preds(obj: object) -> None:
    assert_is_kg_data(obj)
    assert isinstance(obj, KGData), "Object must be a KGData instance"
    pred_fields = ["input_id", "edge_label_index", "edge_label"]
    for field in pred_fields:
        assert hasattr(obj, field), f"KGData missing prediction field '{field}'"
        assert getattr(obj, field) is not None, f"KGData prediction field '{field}' is None"


def has_attr_without_none(kg_data: KGData, attr: str) -> bool:
    value = getattr(kg_data, attr, None)
    return value is not None


class KGData(Data):
    """
    Knowledge Graph Data structure extending PyTorch Geometric's Data class.

    Represents a knowledge graph with nodes, edges, and their associated attributes.
    Supports both sampled subgraphs and full graphs, with optional mappings to
    original graph indices for tracking node and edge origins.

    Attributes:
        x: Node feature tensor of shape [num_nodes, num_node_attrs] containing
            node attributes or features for each node in the graph.

        edge_index: Edge connectivity tensor of shape [2, num_edges] in COO format,
            where edge_index[0] contains source nodes and edge_index[1] contains
            target nodes.

        edge_attr: Edge attribute tensor of shape [num_edges, num_edge_attrs]
            containing attribute information for each edge.

        num_nodes: Integer representing the number of nodes in the graph.

        n_id: Optional tensor of shape [num_nodes] containing global node indices
            mapping. Used when the graph is a sampled subgraph to map local node
            indices back to their original global indices. None for full graphs.

        e_id: Optional tensor of shape [num_edges] containing global edge indices
            mapping. Used when the graph is a sampled subgraph to map local edge
            indices back to their original global indices. None for full graphs.

        num_sampled_nodes: Optional 1D tensor of shape [num_hops] containing the
            number of nodes sampled at each hop during graph sampling. None if
            not applicable.

        num_sampled_edges: Optional 1D tensor of shape [num_hops] containing the
            number of edges sampled at each hop during graph sampling. None if
            not applicable.

        input_id: Optional tensor of shape [batch_size] containing indices of seed
            edges in the batch. These indices reference edges in edge_label_index that
            serve as starting points for link prediction. Used to track which edges
            from the original graph are being used as queries for prediction.
            None if not applicable.

        edge_label_index: Optional tensor of shape [2, batch_size] containing
            source and target node pairs for which link predictions are to be made.
            The first row contains source node indices, and the second row contains
            target node indices. These are the candidate edges that the model should
            predict relations for. None if not applicable.

        edge_label: Optional tensor of shape [batch_size, num_relations] containing
            edge labels in multi-hot encoding format. Each row corresponds to an edge
            in edge_label_index and indicates which relation types are present (1) or
            absent (0) for that edge. Used as ground truth labels during training and
            evaluation. None if not applicable.

        target_timestamps: Optional tensor of shape [batch_size] containing the
            target timestamp for each seed edge in the batch. Used by temporal
            loaders to track which timestamp each prediction target belongs to.
            None if not applicable.

    Example:
        >>> # Create KGData from facts tensor
        >>> facts = torch.tensor([[0, 1, 2], [1, 3, 4], [2, 5, 6]])
        >>> kg_data = KGData.from_facts(facts, num_nodes=7)
        >>> print(kg_data.edge_index.shape)  # [2, 3]
        >>> print(kg_data.edge_attr.shape)   # [3, 1]
    """

    x: Tensor | None
    edge_index: OptTensor
    edge_attr: OptTensor
    n_id: torch.Tensor | None
    e_id: torch.Tensor | None
    num_sampled_nodes: torch.Tensor | None
    num_sampled_edges: torch.Tensor | None
    input_id: torch.Tensor | None
    edge_label_index: torch.Tensor | None
    edge_label: torch.Tensor | None
    neg_edge_label_index: torch.Tensor | None
    relation_labels: torch.Tensor | None
    target_timestamps: torch.Tensor | None

    @classmethod
    def from_facts(
        cls,
        facts: Tensor,
        num_nodes: int,
        domains_mapping_df: pl.DataFrame | None = None,
        num_domains: int | None = None,
        relabel_nodes: bool = True,
        sort: bool = False,
        add_reverse_edges: bool = False,
    ) -> KGData:
        """
        Create a KGData instance from a facts tensor.

        Converts a facts tensor (typically representing triples in a knowledge graph)
        into a KGData object. The facts tensor is expected to have columns representing
        [source_node, relation/attribute, target_node, ...additional_attributes].
        Columns 0 and 2 are used as source and target nodes, while all other columns
        become edge attributes.

                Args:
            facts: Input tensor of shape [num_edges, num_columns] where:
                - Column 0: Source node IDs
                - Column 1: Relation/attribute information (or other edge attributes)
                - Column 2: Target node IDs
                - Columns 3+: Additional edge attributes

            num_nodes: Integer specifying the total number of nodes in the
                graph. Must be at least as large as the maximum node ID in
                the facts tensor.

            domains_mapping_df: Optional Polars DataFrame mapping node IDs to domain
                identifiers. Must have columns including "node_id" and a domain column.
                If provided, node features will be extracted from this mapping.
                Defaults to None.


            num_domains: Optional integer specifying the number of domains. Required
                if domains_mapping_df is provided. If None and domains_mapping_df is
                provided, will be inferred as max(domain_id) + 1. Defaults to None.

            relabel_nodes: If True, remaps node indices in edge_index to be contiguous
                (0, 1, 2, ..., num_nodes-1) and stores original node IDs in n_id.
                If False, preserves original node IDs in edge_index. Defaults to False.

            sort: If True, sort edges by (dst, src). Defaults to False.

            add_reverse_edges: If True, for each fact (s, r, o, ...) append a reverse
                edge (o, s) with the same relation and other attributes, and add a
                third column to edge_attr: 0 for forward edges, 1 for reverse edges.
                Defaults to False.

        Returns:
            KGData instance with:
                - x: Node feature tensor. If domains_mapping_df is provided, contains
                    domain information. Otherwise, contains ones tensor of shape
                    [num_nodes, 1].
                - edge_index: Edge connectivity tensor of shape [2, num_edges] with
                    source and target nodes (potentially relabeled if relabel_nodes=True).
                - edge_attr: Edge attribute tensor of shape [num_edges, num_edge_attrs]
                    containing all columns from facts except columns 0 and 2. When
                    add_reverse_edges is True, edge_attr has an extra column (index 2)
                    with 0 for forward edges and 1 for reverse edges.
                - num_nodes: Number of nodes in the graph.
                - n_id: Tensor of unique node IDs present in the graph. If relabel_nodes
                    is True, this contains the original global node IDs before relabeling.

        """
        if facts.numel() == 0:
            msg = "facts is empty"
            raise ValueError(msg)

        if domains_mapping_df is None and num_domains is not None:
            msg = "num_domains provided without domains_mapping_df"
            raise ValueError(msg)

        device = facts.device

        edge_pairs = facts[:, [0, 2]]  # [N, 2]
        cols = [i for i in range(facts.shape[1]) if i not in [0, 2]]
        edge_attr = facts[:, cols]  # [N, len(cols)]

        edge_index = edge_pairs.t().contiguous()

        if num_nodes < edge_index.max() + 1:
            msg = f"{num_nodes} < edge_index.max() + 1"
            raise IndexError(msg)

        if sort:
            row, col = edge_index  # src, dst

            perm = col * num_nodes + row
            perm = perm.argsort()

            edge_index = edge_index[:, perm]
            edge_attr = edge_attr[perm]

        if add_reverse_edges:
            # Append reverse edges (o, s) with same [r, t, ...] and direction=1
            reverse_edge_index = edge_index[[1, 0], :]  # swap src, dst
            dtype_dir = (
                edge_attr.dtype
                if edge_attr.dtype in (torch.long, torch.int, torch.int32, torch.int64)
                else torch.long
            )
            # Forward edges: add direction column 0
            edge_attr_forward = torch.cat(
                [
                    edge_attr,
                    torch.zeros(edge_attr.size(0), 1, dtype=dtype_dir, device=device),
                ],
                dim=1,
            )
            # Reverse edges: same attr plus direction column 1
            edge_attr_reverse = torch.cat(
                [
                    edge_attr,
                    torch.ones(edge_attr.size(0), 1, dtype=dtype_dir, device=device),
                ],
                dim=1,
            )
            edge_index = torch.cat([edge_index, reverse_edge_index], dim=1)
            edge_attr = torch.cat([edge_attr_forward, edge_attr_reverse], dim=0)
            if sort:
                row, col = edge_index
                perm = (col * num_nodes + row).argsort()
                edge_index = edge_index[:, perm]
                edge_attr = edge_attr[perm]

        full_node_ids = torch.arange(num_nodes, device=device)

        x: Tensor
        if domains_mapping_df is not None:
            num_rows = domains_mapping_df.shape[0]
            if num_rows != num_nodes:
                msg = f"num_rows[{num_rows}] != num_nodes[{num_nodes}]"
                raise IndexError(msg)

            x = KGData.extract_node_domains(
                domains_mapping_df=domains_mapping_df,
                node_ids=full_node_ids,
            ).to(device)

            if num_domains is None:
                num_domains = int(x.max().item()) + 1
        else:
            x = torch.ones((num_nodes, 1), dtype=torch.long, device=device)
        n_id: Tensor | None = None
        if relabel_nodes:
            n_id = cast(Tensor, torch.unique(edge_index))
            num_nodes = cast(int, n_id.max().item()) + 1

            edge_index, edge_attr = subgraph(n_id, edge_index, edge_attr, relabel_nodes=True)
            x = x[n_id]

        return KGData(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=num_nodes,
            n_id=n_id,
        )

    @classmethod
    def extract_node_domains(
        cls,
        domains_mapping_df: pl.DataFrame,
        node_ids: Tensor,
        node_index_col: str = "node_id",
        domain_index_col: str = "domain_id",
    ) -> Tensor:
        return (
            domains_mapping_df.filter(pl.col(node_index_col).is_in(node_ids.tolist()))
            .select(domain_index_col)
            .to_torch(dtype=pl.Int64)
        )

    def has_preds(self) -> bool:
        """
        Check if KGData instance has all prediction fields set.

        Returns True if input_id, edge_label_index, and edge_label are all
        present and not None. Returns False otherwise.

        Returns:
            True if all prediction fields are set, False otherwise.
        """
        return (
            has_attr_without_none(self, "input_id")
            and has_attr_without_none(self, "edge_label_index")
            and has_attr_without_none(self, "edge_label")
        )

    def has_n_id(self) -> bool:
        return has_attr_without_none(self, "n_id")

    def to_facts(self) -> Tensor:
        """
        Convert edge index and edge attributes to a facts tensor.

        Transforms the graph representation (edge_index and edge_attr) into a
        facts tensor where each row represents a (temporal) fact

        Returns:
            Tensor: A tensor of shape [num_facts, 3] or [num_facts, 4].
                - If edge_attr has 1 column: returns [num_facts, 3] with columns
                [head, relation, tail]
                - If edge_attr has 2 columns: returns [num_facts, 4] with columns
                [head, relation, tail, timestamp]
        """
        heads = self.edge_index[0]
        tails = self.edge_index[1]
        relations = self.edge_attr[:, 0]

        if self.edge_attr.shape[1] == 2:
            timestamps = self.edge_attr[:, 1]
            return torch.stack([heads, relations, tails, timestamps], dim=1)
        return torch.stack([heads, relations, tails], dim=1)

    def print_summary(self, max_nodes: int = 10, max_edges: int = 10) -> None:
        if hasattr(self, "x") and self.x is not None:
            print(f"x:\n{self.x[:max_nodes]}")
        print(f"\nedge_index:\n{self.edge_index[:, :max_edges]}")
        print(f"\nedge_attr:\n{self.edge_attr[:max_edges]}")
        if hasattr(self, "n_id") and self.n_id is not None:
            print(f"\nn_id:\n{self.n_id[:max_nodes]}")
        if hasattr(self, "e_id") and self.e_id is not None:
            print(f"\ne_id:\n{self.e_id[:max_edges]}")
        if hasattr(self, "input_id") and self.input_id is not None:
            print(f"\ninput_id:\n{self.input_id[:max_edges]}")
        if hasattr(self, "edge_label_index") and self.edge_label_index is not None:
            print(f"\nedge_label_index:\n{self.edge_label_index[:, :max_edges]}")
        if hasattr(self, "edge_label") and self.edge_label is not None:
            print(f"\nedge_label:\n{self.edge_label[:max_edges]}")

        print(self)

    def get_metadata(self) -> dict[str, Any]:
        num_domains: int | None = None
        has_domains = False
        if self.x is not None:
            num_domains = self.x.unique().numel()
            has_domains = True

        num_relations = self.edge_attr[:, 0].unique().numel()
        num_timestamps = self.edge_attr[:, 1].unique().numel()
        return {
            "num_nodes": int(self.num_nodes),
            "num_edges": int(self.edge_index.shape[1]),
            "num_relations": int(self.edge_attr[:, 0].max().item() + 1),
            "num_domains": num_domains,
            "has_domains": has_domains,
            "has_n_id": self.has_n_id(),
            "has_preds": self.has_preds(),
            "x_shape": tuple(self.x.shape) if self.x is not None else None,
            "edge_attr_shape": tuple(self.edge_attr.shape),
            "num_relations_in_data": num_relations,
            "num_timestamps": num_timestamps,
        }
