"""SeGAL inference wrapper for deployment."""

from __future__ import annotations

from torch import Tensor, nn

from hakken_models.core.entities.kg_data import KGData

from .base import SeGAL
from .schemas import ScoreBatchOutput


class SeGALInferenceWrapper(nn.Module):
    """Deployment-ready wrapper bundling SeGAL with node and relation embeddings.

    Exposes :meth:`score_batch` with no embedding arguments — useful for
    evaluation, serving, and single-object deployment.
    """

    def __init__(
        self,
        segal: SeGAL,
        node_embeddings: Tensor,
        relation_embeddings: Tensor,
    ) -> None:
        super().__init__()
        self.segal = segal
        self.register_buffer("node_embeddings", node_embeddings)
        self.register_buffer("relation_embeddings", relation_embeddings)

    def score_batch(self, batch: KGData) -> ScoreBatchOutput:
        """Score a batch of positives and negatives.

        Args:
            batch: KGData with edge_label_index, neg_edge_label_index.

        Returns:
            ScoreBatchOutput with pos_scores [B] and neg_scores [B, K].
        """
        return self.segal.score_batch(
            batch,
            self.node_embeddings,
            self.relation_embeddings,
        )
