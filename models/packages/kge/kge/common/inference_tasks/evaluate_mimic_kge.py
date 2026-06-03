from __future__ import annotations

from pathlib import Path

import hydra
import pandas as pd
from datasets.common.constants import DataSplits
from loguru import logger
from omegaconf import DictConfig

from kge.common.actions.gnnkge_loader_action import GNNKGEExperimentData
from kge.evaluator.mimic_kge import MimicKGEEvaluator


def run_evaluate_mimic_kge(config: DictConfig, bundle: GNNKGEExperimentData) -> None:
    model = bundle.model
    trained_kge = bundle.trained_kge
    if trained_kge is None:
        msg = "Trained KGE model must be provided for MimicKGE evaluation."
        raise ValueError(msg)
    data_processor = bundle.data_processor

    evaluator: MimicKGEEvaluator = hydra.utils.instantiate(config.evaluator)

    evaluator.set_model(model)
    evaluator.init()

    data_split_list = [DataSplits.TRAIN, DataSplits.VALID, DataSplits.TEST]

    list_df = []
    for data_split in data_split_list:
        subgraph_split = [DataSplits.TRAIN]
        if data_split == DataSplits.TEST:
            subgraph_split.append(DataSplits.VALID)
        loader = data_processor.get_mimic_kge_data_loader(
            split=data_split,
            trained_kge=trained_kge,
            subgraph_split=subgraph_split,
            shuffle=False,
        )

        loader.load_scaler()

        metrics_df_i = evaluator.evaluate_from_dataloader(data_loader=loader, device=config.device)

        metrics_df_i["split"] = data_split.value
        list_df.append(metrics_df_i)

    metrics_df = pd.concat(list_df, ignore_index=True)

    metrics_path = Path(config.output_dir) / "evaluation_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, sep="\t")
    logger.info(f"Evaluation metrics saved to {metrics_path}")
