from __future__ import annotations

from pathlib import Path

import hydra
import pandas as pd
from datasets.common.constants import DataSplits
from loguru import logger
from omegaconf import DictConfig

from kge.common.actions.kge_loader_action import KGEExperimentData
from kge.common.exceptions import SplitsError
from kge.evaluator import KGEEvaluator
from kge.negative_sampler import NegativeSamplerI


def run_evaluate(config: DictConfig, bundle: KGEExperimentData) -> None:
    model = bundle.model
    data_processor = bundle.data_processor

    evaluator: KGEEvaluator = hydra.utils.instantiate(config.evaluator)
    negative_sampler: NegativeSamplerI = hydra.utils.instantiate(config.negative_sampler)

    evaluator.set_model(model)

    evaluator.init(data_processor.kg)
    negative_sampler.set_up(data_processor.kg)

    train_dataset = data_processor.get_dataset(split=DataSplits.TRAIN)

    model.to_device(config.device)

    model.fit_score_scaler_from_dataset(
        dataset=train_dataset,
        negative_sampler=negative_sampler,
        loader_kwargs=evaluator.config.loader_kwargs,
        json_path=config.score_scaler_path,
    )
    model.to_device(config.device)

    num_triples = 2048 * 5

    metrics_df_list = []

    data_split_list = [DataSplits.TRAIN, DataSplits.VALID, DataSplits.TEST]

    for data_split in data_split_list:
        try:
            dataset = data_processor.get_dataset(split=data_split, num_triples=num_triples)
        except SplitsError:
            continue

        sup_dataset = data_processor.get_supervised_dataset(
            split=data_split, num_triples=num_triples
        )

        metrics_df_i = evaluator.evaluate(
            dataset=dataset,
            sup_dataset=sup_dataset,
            device=config.device,
            split_name=data_split.value,
        )
        metrics_df_list.append(metrics_df_i)

    metrics_df = pd.concat(metrics_df_list, ignore_index=True)

    metrics_path = Path(config.output_dir) / "evaluation_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, sep="\t")
    logger.info(f"Evaluation metrics saved to {metrics_path}")
