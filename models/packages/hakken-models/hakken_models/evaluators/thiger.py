from hakken_models.core.entities.kg_data_with_preds import KGDataWithPreds
from hakken_models.models.thiger import THiGER

from .base import HakkenModelEvaluator


class THiGEREvaluator(HakkenModelEvaluator[THiGER, KGDataWithPreds]):
    def update_from_batch(self, model: THiGER, batch: KGDataWithPreds) -> None:
        """Update metrics based on a single batch.

        Args:
            model: The model to evaluate
            batch: A single batch of data
        """
        device = next(model.parameters()).device
        batch = batch.to(device)
        entity_pair_batch = batch.edge_label_index.t().contiguous()
        model.set_context_temporal_kg(batch)
        logits = model.compute_logits(entity_pair_batch)
        targets = batch.edge_label

        for metric_bundle in self.metrics.values():
            metric_bundle.instance.update(logits, targets)
