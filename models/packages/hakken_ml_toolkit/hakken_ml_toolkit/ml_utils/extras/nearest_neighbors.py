from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812

if TYPE_CHECKING:
    from hakken_ml_toolkit.ml_utils.extras.domain import FloatTensor1D, LongTensor1D


class NearestNeighbors:
    @staticmethod
    def embeddings_l2(
        query_embeddings: torch.Tensor, embeddings: torch.Tensor, top_k: int = 1
    ) -> tuple[FloatTensor1D, LongTensor1D]:
        if top_k != 1:
            raise NotImplementedError

        diff = query_embeddings.unsqueeze(1) - embeddings.unsqueeze(0)
        distances = torch.norm(diff, dim=2)

        # Find the indices of the minimum distances
        distances, indices = distances.min(dim=1)

        return distances, indices

    @staticmethod
    def embeddings_cosine(
        query_embeddings: torch.Tensor, embeddings: torch.Tensor, top_k: int = 1
    ) -> tuple[FloatTensor1D, LongTensor1D]:
        if top_k != 1:
            raise NotImplementedError

        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        query_embeddings_norm = F.normalize(query_embeddings, p=2, dim=1)

        similarity = torch.mm(query_embeddings_norm, embeddings_norm.t())

        similarity, indices = similarity.max(dim=1)

        return similarity, indices
