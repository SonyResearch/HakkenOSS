import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm

from kge.data_loaders.mimic_kge import MimicKGEDataLoader
from kge.data_loaders.mimic_kge.loader import BatchType
from kge.evaluator.config import MimicKGEEvaluatorConfig
from kge.evaluator.utils import KGEEvalUtils
from kge.models.gnn import GNNKGE


class MimicKGEEvaluator:
    def __init__(
        self,
        config: MimicKGEEvaluatorConfig,
    ):
        """Evaluator for MimicKGE.

        Attributes:
            config: Configuration object containing evaluation settings and metrics.
            metrics: List of metric instances created from the configuration.
        """
        self.config = config

        self._is_enabled = config.enable

        self._model: GNNKGE | None = None

    def set_model(self, model: GNNKGE) -> None:
        self._model = model

    def init(self) -> None:
        """Initialize the evaluator by creating metrics from configuration.

        This method sets up all metrics specified in the configuration.
        If the evaluator is disabled, this method returns early without
        performing any initialization.

        Note:
            Must be called after setting the model and before evaluation.
        """
        if not self._is_enabled:
            return
        logger.info("Initializing KGEEvaluator with KnowledgeGraph")

        logger.info("Creating metrics...")
        self.metrics, _targets = KGEEvalUtils.create_metrics_from_config(
            metrics_config=self.config.metrics
        )

        if not self._is_enabled:
            return

    def enable(self) -> None:
        self._is_enabled = True

    def disable(self) -> None:
        self._is_enabled = False

    def is_enabled(self) -> bool:
        return self._is_enabled

    def reset(self) -> None:
        """Reset all metrics to their initial state.

        This method should be called before starting a new evaluation
        to ensure metrics don't accumulate values from previous runs.
        If the evaluator is disabled, this method returns early.
        """
        if not self._is_enabled:
            return
        for metric in self.metrics:
            metric.metric_instance.reset()

    @torch.no_grad()
    def evaluate_from_dataloader(
        self,
        data_loader: MimicKGEDataLoader,
        device: str | torch.device,
    ) -> pd.DataFrame:
        """Evaluate the model on all batches from the data loader.

        This method processes all batches in the data loader, computes model
        scores, and updates metrics accordingly. The model is set to evaluation
        mode and gradients are disabled for efficiency.

        Args:
            data_loader: MimicKGE data loader containing evaluation batches.
            device: Device to run evaluation on (e.g., 'cuda', 'cpu', or torch.device).

        Returns:
            DataFrame containing computed metrics with columns:
                - name: metric name
                - value: computed metric value
                - additional parameter columns from metric configuration

        Note:
            Returns empty DataFrame if evaluator is disabled.

        """
        if not self._is_enabled:
            return pd.DataFrame()

        self.reset()
        self.to_device(device)
        self._model.eval()

        batch: BatchType
        for batch in tqdm(data_loader, desc="Processing batches", total=len(data_loader)):
            pred_subgraph = batch[0].to(device)
            target_pos = batch[3].to(device)

            scores_pos = self._model.score(pred_subgraph)

            self.update(scores=scores_pos, targets=target_pos)

        return self.compute_metrics()

    def update(self, scores: torch.Tensor, targets: torch.Tensor) -> None:
        """Update all metrics with new predictions and targets.

        This method updates each metric instance with the provided scores
        and targets, mapping the arguments according to each metric's
        configuration.

        Args:
            scores: Model prediction scores as a tensor.
            targets: Ground truth target values as a tensor.

        Note:
            The method automatically maps available variables (scores, targets)
            to each metric's expected update arguments based on the metric's
            update_args_mapping configuration.
        """

        for metric in self.metrics:
            variables = {
                "targets": targets,
                "scores": scores,
            }
            update_args = {
                key: variables[var_name]
                for key, var_name in metric.update_args_mapping.items()
                if var_name in variables
            }

            metric.metric_instance.update(**update_args)

    def to_device(self, device: str | torch.device) -> None:
        for metric in self.metrics:
            if hasattr(metric.metric_instance, "to"):
                metric.metric_instance = metric.metric_instance.to(device)

        if self._model is not None:
            self._model.to(device)

    def compute_metrics(self) -> pd.DataFrame:
        if not self._is_enabled:
            return pd.DataFrame()

        data: list[dict] = []
        for metric in self.metrics:
            metric_name = metric.name
            value = metric.metric_instance.compute()
            if isinstance(value, torch.Tensor):
                value = value.item()
            data.append({"name": metric_name, **metric.parameters, "value": value})

        return pd.DataFrame(data)
