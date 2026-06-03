"""SeGAL evaluator for entity-ranking metrics (mean rank)."""

import torch

from hakken_models.core.entities.kg_data import KGData
from hakken_models.models.segal import SeGALInferenceWrapper

from .base import HakkenModelEvaluator


class SeGALEvaluator(HakkenModelEvaluator[SeGALInferenceWrapper, KGData]):
    """Evaluator for SeGAL using pos_scores vs neg_scores (mean rank)."""

    def update_from_batch(self, model: SeGALInferenceWrapper, batch: KGData) -> None:
        """Update metrics based on a single batch."""
        device = next(model.parameters()).device
        batch = batch.to(device)

        with torch.no_grad():
            out = model.score_batch(batch)

        for metric_bundle in self.metrics.values():
            metric_bundle.instance.update(out.pos_scores, out.neg_scores)
