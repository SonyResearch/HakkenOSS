from __future__ import annotations

from typing import Any, cast

from torch import Tensor
from torch_geometric.loader import LinkNeighborLoader

from hakken_models.core.entities.kg_data import KGData
from hakken_models.core.entities.kg_data_with_preds import KGDataWithPreds


class KGLinkNeighborLoader(LinkNeighborLoader):
    def __init__(
        self,
        data: KGData,
        num_neighbors: list[int],
        edge_label_index: Tensor | None = None,
        edge_label: Tensor | None = None,
        batch_size: int = 1,
        shuffle: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the KG link neighbor data loader.

        Args:
            data: Knowledge graph data without original node indexes.
            num_neighbors: Number of neighbors to sample at each layer.
            batch_size: Number of samples per batch. Defaults to 1.
            edge_label_index: Edge indices for link prediction labels.
            edge_label: Edge labels for link prediction.
            shuffle: Whether to shuffle the data. Defaults to True.

        Raises:
            ValueError: If data contains original node indexes.
        """

        self.num_neighbors = num_neighbors

        if data.has_n_id():
            msg = "data must not contain the original indexes"
            raise ValueError(msg)

        super().__init__(
            data=data,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            edge_label_index=edge_label_index,
            edge_label=edge_label,
            neg_sampling=None,
            is_sorted=False,
            filter_per_worker=False,
            shuffle=shuffle,
            **kwargs,
        )

    def __call__(self, index: Tensor | list[int]) -> KGDataWithPreds:
        raw_batch = super().__call__(index)
        raw_batch.edge_label_index = raw_batch.n_id[raw_batch.edge_label_index]
        return cast(KGDataWithPreds, raw_batch)
