from torch import Tensor

from hakken_models.models.kge import KGE

from .base import HakkenModelEvaluator

TensorList = list[Tensor]


class KGEEvaluator(HakkenModelEvaluator[KGE, TensorList]):
    def compute_metric_inputs(
        self, model: KGE, facts_tensor: Tensor
    ) -> dict[str, tuple[Tensor, Tensor]]:
        inputs = {}

        for mode in self.unique_prediction_modes:
            if mode == "object":
                scores = model.score_objects(head=facts_tensor[:, 0], relation=facts_tensor[:, 1])
                targets = facts_tensor[:, 2].long()
                inputs[mode] = (scores, targets)
            elif mode == "subject":
                scores = model.score_subjects(relation=facts_tensor[:, 1], tail=facts_tensor[:, 2])
                targets = facts_tensor[:, 0].long()
                inputs[mode] = (scores, targets)
            elif mode == "relation":
                scores = model.score_relations(head=facts_tensor[:, 0], tail=facts_tensor[:, 2])
                targets = facts_tensor[:, 1].long()
                inputs[mode] = (scores, targets)
            else:
                raise ValueError(f"Unsupported prediction mode '{mode}'")

        return inputs

    def update_from_batch(self, model: KGE, batch: list[Tensor]) -> None:
        """Update metrics based on a single batch.

        Args:
            model: The model to evaluate
            batch: A single batch of data
        """
        device = next(model.parameters()).device
        facts_tensor = batch[0].to(device)

        inputs_dict = self.compute_metric_inputs(model, facts_tensor)

        for metric_bundle in self.metrics.values():
            metric_config = metric_bundle.config
            metric = metric_bundle.instance
            prediction_mode = metric_config.prediction_mode

            if prediction_mode is None:
                raise ValueError(
                    f"Metric '{metric_config.name}' does not specify a prediction mode. "
                    "This is required to determine which inputs to use for the metric."
                )
            scores, targets = inputs_dict[prediction_mode]
            if metric_bundle.relation_id is not None:
                mask = facts_tensor[:, 1] == metric_bundle.relation_id
                if mask.sum() == 0:
                    continue  # No examples for this relation in the batch, skip metric update

                metric.update(scores[mask], targets[mask])
            else:
                metric.update(scores, targets)
