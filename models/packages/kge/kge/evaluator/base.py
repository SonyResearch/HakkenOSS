import math

import pandas as pd
import torch
from hakken_ml_toolkit.ml_base_structures import KnowledgeGraph
from hakken_ml_toolkit.ml_utils.extras import FactBatchUtils
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from kge.common.constants import TargetType
from kge.common.types import FloatTensor2D, LongTensor2D
from kge.evaluator.config import KGEEvaluatorConfig
from kge.evaluator.entities import EvaluationMetric
from kge.evaluator.utils import KGEEvalUtils
from kge.models.base import KGEI
from kge.triple_filterer.hard_filtering import (
    HardTripleFilter,
    HardTripleFilterConfig,
)


class KGEEvaluator:
    def __init__(
        self,
        config: KGEEvaluatorConfig,
        kg: KnowledgeGraph | None = None,
    ):
        """
        Initializes the KGEEvaluator with the provided configuration.

        Args:
            config (KGEEvaluatorConfig): The configuration containing the list of metrics.
        """
        self.config = config

        self.filterer: dict[TargetType, HardTripleFilter] = {}

        self._is_enabled = config.enable

        self._model: KGEI | None = None

        self.kg = kg

    def set_model(self, model: KGEI) -> None:
        self._model = model

    def init(self, kg: KnowledgeGraph) -> None:
        if not self._is_enabled:
            return
        logger.info("Initializing KGEEvaluator with KnowledgeGraph")

        logger.info("Creating ranking metrics...")
        ranking_metrics, targets = KGEEvalUtils.create_metrics_from_config(
            metrics_config=self.config.ranking_metrics
        )
        self.ranking_metrics: list[EvaluationMetric] = ranking_metrics
        self.targets: set[TargetType] = targets

        logger.info("Creating relation classification metrics...")
        clf_metrics, _ = KGEEvalUtils.create_metrics_from_config(
            metrics_config=self.config.relation_clf_metrics, num_labels=kg.num_relations
        )

        self.relation_clf_metrics: list[EvaluationMetric] = clf_metrics

        if not self._is_enabled:
            return
        self.kg = kg
        if self.config.filter_list is not None:
            logger.info("Setting filterers...")
            for target in self.targets:
                self.filterer[target] = HardTripleFilter(
                    kg=kg,
                    config=HardTripleFilterConfig(
                        target=target, filter_list=self.config.filter_list
                    ),
                )

    def enable(self) -> None:
        self._is_enabled = True

    def disable(self) -> None:
        self._is_enabled = False

    def is_enabled(self) -> bool:
        return self._is_enabled

    def reset(self) -> None:
        """
        Resets all metrics to their initial state.
        """
        if not self._is_enabled:
            return
        for ranking_metric in self.ranking_metrics:
            ranking_metric.metric_instance.reset()

        for clf_metric in self.relation_clf_metrics:
            clf_metric.metric_instance.reset()

    def update_relation_clf(self, so_batch: LongTensor2D, relations_batch: FloatTensor2D) -> None:
        scores = self._model.score_relations(so_batch)
        norm_scores = self._model.normalize_scores(scores)

        for metric in self.relation_clf_metrics:
            variables = {
                "relations": relations_batch,
                "scores": norm_scores,
            }
            update_args = {
                key: variables[var_name]
                for key, var_name in metric.update_args_mapping.items()
                if var_name in variables
            }

            metric.metric_instance.update(**update_args)

    def update(self, sro_batch: LongTensor2D, scores_dict: dict[TargetType, FloatTensor2D]) -> None:
        """
        Updates evaluation metrics based on model predictions for a batch of triples.

        This method should be called for each batch during evaluation to accumulate
        metric values across the entire dataset.

        Args:
            sro_batch (LongTensor2D): Batch of triples in the form of [subject, relation, object]
                with shape [batch_size, 3].
            scores (FloatTensor2D): Prediction scores from the model with shape
                [batch_size, num_entities]. Each row contains scores for all possible
                entities that could complete the triple, given the subject and relation.
            target (TargetType | None, optional): The entity type to target for evaluation
                (subject, relation, or object). If None, defaults to the first target type
                specified in the configuration. Defaults to None.
            inplace (bool, optional): If True, updates the metrics in place. If False,
                creates a copy of the metrics before updating. Defaults to True.

        Returns:
            None: The metrics are updated internally and can be retrieved calling compute.
        """
        if not self._is_enabled:
            return

        for metric in self.ranking_metrics:
            targets = KGEEvalUtils.get_targets_from_sro_batch(sro_batch, metric.target)

            variables = {
                "targets": targets,
                "scores": scores_dict[metric.target],
            }
            update_args = {
                key: variables[var_name]
                for key, var_name in metric.update_args_mapping.items()
                if var_name in variables
            }

            metric.metric_instance.update(**update_args)

    @torch.no_grad()
    def get_scores_dict(self, sro_batch: LongTensor2D) -> dict[TargetType, FloatTensor2D]:
        scores_dict: dict[TargetType, FloatTensor2D] = {}

        for target in self.targets:
            if target == TargetType.SUBJECT:
                ro_batch_i = FactBatchUtils.to_ro_batch(sro_batch)
                scores = self._model.score_subjects(ro_batch_i)

            elif target == TargetType.RELATION:
                so_batch_i = FactBatchUtils.to_so_batch(sro_batch)
                scores = self._model.score_relations(so_batch_i)

            elif target == TargetType.OBJECT:
                sr_batch_i = FactBatchUtils.to_sr_batch(sro_batch)
                scores = self._model.score_objects(sr_batch_i)

            if target in self.filterer:
                scores_dict[target] = self.filterer[target].compute_scores(
                    sro_batch=sro_batch, scores=scores, inplace=False
                )
            else:
                scores_dict[target] = scores
        return scores_dict

    @torch.no_grad()
    def update_in_batches(self, sro_batch: LongTensor2D, batch_size: int = 128) -> None:
        if not self._is_enabled:
            return
        sro_batch_size = sro_batch.shape[0]

        num_batches = math.ceil(sro_batch_size / batch_size)
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            sro_batch_i = sro_batch[start_idx:end_idx]

            scores_dict = self.get_scores_dict(sro_batch_i)

            self.update(sro_batch=sro_batch_i, scores_dict=scores_dict)

    def evaluate_from_supervised_dataset(
        self,
        sup_dataset: Dataset,
        device: str | torch.device,
    ) -> pd.DataFrame:
        if not self._is_enabled:
            return pd.DataFrame()

        data_loader = DataLoader(
            sup_dataset,
            **self.config.loader_kwargs,
        )

        self.reset()
        self.to_device(device)

        batch_size = 128

        for batch in tqdm(data_loader, desc="Processing batches", total=len(data_loader)):
            so_batch: LongTensor2D = batch[0].to(device)
            relations_batch: LongTensor2D = batch[1].to(device)

            so_batch_size = so_batch.shape[0]
            num_batches = math.ceil(so_batch_size / batch_size)
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = (i + 1) * batch_size
                so_batch_i = so_batch[start_idx:end_idx]
                relations_batch_i = relations_batch[start_idx:end_idx]

                self.update_relation_clf(
                    so_batch=so_batch_i,
                    relations_batch=relations_batch_i,
                )

        return self.compute_clf_metrics()

    def to_device(self, device: str | torch.device) -> None:
        for metric in self.ranking_metrics + self.relation_clf_metrics:
            if hasattr(metric.metric_instance, "to"):
                metric.metric_instance = metric.metric_instance.to(device)

        if self._model is not None:
            self._model.to_device(device)

    def evaluate_from_dataset(self, dataset: Dataset, device: str | torch.device) -> pd.DataFrame:
        if not self._is_enabled:
            return pd.DataFrame()

        data_loader = DataLoader(
            dataset,
            **self.config.loader_kwargs,
        )

        return self.evaluate_from_data_loader(data_loader=data_loader, device=device)

    def evaluate_from_data_loader(
        self, data_loader: DataLoader, device: str | torch.device
    ) -> pd.DataFrame:
        self.reset()
        self.to_device(device)

        for batch in tqdm(data_loader, desc="Processing batches", total=len(data_loader)):
            sro_batch: LongTensor2D = batch[0]
            self.update_in_batches(sro_batch=sro_batch.to(device), batch_size=512)

        return self.compute_metrics()

    def evaluate(
        self,
        dataset: Dataset | None = None,
        sup_dataset: Dataset | None = None,
        device: str | torch.device = "gpu",
        split_name: str = "default",
    ) -> pd.DataFrame:
        df_list = []
        if dataset is not None:
            ranking_df = self.evaluate_from_dataset(dataset=dataset, device=device)
            ranking_df["split"] = split_name
            df_list.append(ranking_df)

        if sup_dataset is not None:
            clf_df = self.evaluate_from_supervised_dataset(sup_dataset=sup_dataset, device=device)
            clf_df["split"] = split_name
            df_list.append(clf_df)

        if len(df_list) == 0:
            msg = "Both `dataset` and `sup_dataset` are None. Please provide at least one."
            raise ValueError(msg)
        return pd.concat(df_list, ignore_index=True)

    def compute_clf_metrics(self) -> pd.DataFrame:
        if not self._is_enabled:
            return pd.DataFrame()

        data: list[dict] = []
        for metric in self.relation_clf_metrics:
            metric_name = metric.name
            value = metric.metric_instance.compute()
            if isinstance(value, torch.Tensor):
                value = value.item()
            data.append({"name": metric_name, **metric.parameters, "value": value})

        return pd.DataFrame(data)

    def compute_metrics(self) -> pd.DataFrame:
        if not self._is_enabled:
            return pd.DataFrame()

        data: list[dict] = []
        for metric in self.ranking_metrics:
            metric_name = metric.name
            value = metric.metric_instance.compute()
            if isinstance(value, torch.Tensor):
                value = value.item()
            data.append({"name": metric_name, **metric.parameters, "value": value})

        return pd.DataFrame(data)
