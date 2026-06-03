from typing import NamedTuple, cast

import torch
from torch import Tensor
from torch_geometric.utils.map import map_index

from .kg_data import KGData


class TimestampSubgraphData(NamedTuple):
    """
    Data structure for timestamp-filtered subgraph information.

    Attributes:
        node_data: Node feature tensor of shape [num_nodes, num_node_features]
            containing features for all nodes present in the timestamp subgraph.
        n_id: Tensor of shape [num_nodes] containing the original global node IDs
            present in the subgraph (sorted in ascending order).
        edge_index: Edge connectivity tensor of shape [2, num_edges] in COO format.
            If relabel_nodes=True, indices are remapped to [0, num_nodes-1].
            If relabel_nodes=False, indices are original global node IDs.
        edge_attr: Edge attribute tensor of shape [num_edges, num_edge_features]
            containing attributes for edges at the specified timestamp.
    """

    node_data: Tensor
    n_id: Tensor
    edge_index: Tensor
    edge_attr: Tensor


class TemporalKGData:
    """Manages temporal knowledge graph data with timestamp-based filtering.

    This class provides functionality to extract subgraphs filtered by timestamps
    and convert between local and global node ID representations.
    """

    def __init__(self, data: KGData, edge_attr_timestamp_col: int = 1) -> None:
        """Initialize temporal knowledge graph data handler.

        Args:
            data: Knowledge graph data containing nodes, edges, and attributes.
            edge_attr_timestamp_col: Column index in edge_attr tensor containing
                timestamp information. Defaults to 1.
        """
        self.data = data
        self.edge_timestamps = self.data.edge_attr[:, edge_attr_timestamp_col]
        self.unique_timestamps = torch.unique(self.edge_timestamps).tolist()

        self.device = self.data.edge_index.device

        is_sorted = True
        if self.data.has_n_id():
            is_sorted = cast(bool, torch.all(torch.diff(self.data.n_id) >= 0).item())

        if not is_sorted:
            self.n_id_sorted, self.perm = torch.sort(self.data.n_id)
        elif self.data.has_n_id():
            self.n_id_sorted = self.data.n_id
            self.perm = torch.arange(len(self.data.n_id), device=self.device)
        else:
            self.n_id_sorted = torch.unique(self.data.edge_index)
            self.perm = torch.arange(len(self.n_id_sorted), device=self.device)

    @property
    def num_entities(self) -> int:
        return self.data.n_id.size(0) if self.has_n_id() else self.data.x.size(0)

    @property
    def max_index(self) -> int:
        return self.data.edge_index.max().item()

    def list_timestamps(self) -> list[int]:
        """Return list of unique timestamps present in the graph.

        Returns:
            Sorted list of unique timestamp values found in edge attributes.
        """
        return self.unique_timestamps

    def get_node_data(
        self, node_ids: Tensor, is_global: bool = False, safe: bool = False
    ) -> Tensor:
        """Retrieve node feature data for specified node IDs.

        Args:
            node_ids: Tensor of node IDs to retrieve features for.
            is_global: If True, treats node_ids as global IDs and
                converts them to local IDs. If False, treats node_ids as
                local IDs. Defaults to False.
            safe: If True, validates node IDs and raises IndexError for
                invalid IDs. Defaults to False.

        Returns:
            Tensor of shape [num_nodes, num_features] containing node features
            for the requested nodes.

        Raises:
            IndexError: If safe=True and invalid node IDs are provided.
        """
        node_ids_local = self.global_to_local(node_ids, safe=safe) if is_global else node_ids

        if safe:
            max_local = self.data.x.size(0) - 1
            invalid_mask = (node_ids_local < 0) | (node_ids_local > max_local)
            if invalid_mask.any():
                invalid_ids = node_ids_local[invalid_mask].unique().tolist()
                msg = (
                    f"Invalid local node IDs: {invalid_ids}. "
                    f"Valid local ID range: [0, {max_local}] "
                    f"(total nodes: {self.data.x.size(0)})"
                )
                raise IndexError(msg)

        return self.data.x[node_ids_local]

    def get_timestamp_data(
        self, timestamp_idx: int, relabel_nodes: bool = True
    ) -> TimestampSubgraphData:
        """Extract subgraph data filtered by a specific timestamp.

        Args:
            timestamp_idx: Timestamp value to filter edges by.
            relabel_nodes: If True, remaps node indices to [0, num_nodes-1].
                If False, preserves original global node IDs. Defaults to True.

        Returns:
            TimestampSubgraphData containing node features, node IDs,
            edge connectivity, and edge attributes for the
            timestamp-filtered subgraph.

        Raises:
            ValueError: If timestamp_idx is not found in the graph or if no
                edges exist for the specified timestamp.
        """
        if timestamp_idx not in self.unique_timestamps:
            raise ValueError(f"{timestamp_idx} is not valid!")
        edge_mask = self.edge_timestamps == timestamp_idx
        if not edge_mask.any():
            raise ValueError(f"No edges found for timestamp_idx={timestamp_idx}. ")

        edge_index = self.data.edge_index[:, edge_mask]
        edge_attr = self.data.edge_attr[edge_mask]

        n_id_local: Tensor = torch.unique(edge_index.flatten())
        node_data = self.data.x[n_id_local]

        if relabel_nodes:
            edge_index, _ = map_index(
                edge_index.view(-1),
                n_id_local,
                max_index=edge_index.max() + 1,
                inclusive=True,
            )
            edge_index = edge_index.view(2, -1)

        n_id = self.local_to_global(n_id_local)
        return TimestampSubgraphData(
            node_data=node_data,
            n_id=n_id,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

    def has_n_id(self) -> bool:
        return self.data.has_n_id()

    def local_to_global(
        self, local_ids: Tensor, n_id: Tensor | None = None, safe: bool = False
    ) -> Tensor:
        """Convert local node IDs to global node IDs.

        Args:
            local_ids: Tensor of local node IDs (indices into n_id
                array).
            n_id: Optional node ID mapping tensor. If None, uses the
                graph's default n_id mapping. Defaults to None.
            safe: If True, validates local IDs and raises IndexError
                for invalid IDs. Defaults to False.

        Returns:
            Tensor of global node IDs corresponding to the input local IDs.

        Raises:
            IndexError: If safe=True and invalid local IDs are provided.
        """
        if n_id is None:
            if self.has_n_id():
                n_id = self.data.n_id
            else:
                return local_ids

        if safe:
            max_local = n_id.size(0) - 1
            invalid_mask = (local_ids < 0) | (local_ids > max_local)

            if invalid_mask.any():
                invalid_ids = local_ids[invalid_mask].unique().tolist()
                msg = (
                    f"Invalid local IDs found: {invalid_ids}. "
                    f"Valid local ID range: [0, {max_local}] "
                    f"(total nodes in subgraph: {n_id.size(0)})"
                )
                raise IndexError(msg)

        return n_id[local_ids]

    def _global_to_local_sorted(
        self, global_ids: Tensor, n_id_sorted: Tensor, perm: Tensor, safe: bool = False
    ) -> Tensor:
        """Convert global node IDs to local IDs using sorted n_id array.

        Uses binary search for efficient lookup when n_id is sorted.

        Args:
            global_ids: Tensor of global node IDs to convert.
            n_id_sorted: Sorted tensor of global node IDs.
            perm: Permutation tensor mapping sorted indices to original
                indices.
            safe: If True, validates that all global IDs exist and
                raises ValueError for missing IDs. Defaults to False.

        Returns:
            Tensor of local node IDs corresponding to the input global
            IDs.

        Raises:
            ValueError: If safe=True and any global IDs are not found
                in n_id_sorted.
        """
        idx_sorted = torch.searchsorted(n_id_sorted, global_ids)

        if safe:
            out_of_bounds = idx_sorted >= len(n_id_sorted)
            valid_mask = ~out_of_bounds
            if valid_mask.any():
                exact_matches = n_id_sorted[idx_sorted[valid_mask]] == global_ids[valid_mask]
                valid_mask[valid_mask.clone()] = exact_matches

            if idx_sorted.min() == 0:
                too_small = global_ids < n_id_sorted[0]
                if too_small.any():
                    valid_mask = valid_mask & ~too_small
            if not valid_mask.all():
                invalid_ids = global_ids[~valid_mask].unique().tolist()
                raise ValueError(f"Global IDs not found: {invalid_ids}")
        return perm[idx_sorted]

    def _global_to_local_unsorted(
        self, global_ids: Tensor, n_id: Tensor, safe: bool = False
    ) -> Tensor:
        n_id_sorted, perm = torch.sort(n_id)
        return self._global_to_local_sorted(global_ids, n_id_sorted, perm, safe)

    def global_to_local(
        self, global_ids: Tensor, n_id: Tensor | None = None, safe: bool = False
    ) -> Tensor:
        """Convert global node IDs to local node IDs.

        Automatically selects the appropriate conversion method based on
        whether n_id is sorted and whether a custom n_id mapping is
        provided.

        Args:
            global_ids: Tensor of global node IDs to convert.
            n_id: Optional node ID mapping tensor. If None, uses the
                graph's precomputed sorted mapping. Defaults to None.
            safe: If True, validates that all global IDs exist and
                raises ValueError for missing IDs. Defaults to False.

        Returns:
            Tensor of local node IDs corresponding to the input global IDs.

        Raises:
            ValueError: If safe=True and any global IDs are not found.
        """
        if n_id is None:
            if self.has_n_id():
                # Use precomputed sorted version from __init__
                return self._global_to_local_sorted(global_ids, self.n_id_sorted, self.perm, safe)
            return global_ids
        # Check if provided n_id is already sorted
        is_n_id_sorted = torch.all(torch.diff(n_id) >= 0)
        if is_n_id_sorted:
            perm = torch.arange(len(n_id), device=n_id.device)
            return self._global_to_local_sorted(global_ids, n_id, perm, safe)
        return self._global_to_local_unsorted(global_ids, n_id, safe)
