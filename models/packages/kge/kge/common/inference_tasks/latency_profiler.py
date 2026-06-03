from __future__ import annotations

from pathlib import Path

import pandas as pd
from hakken_ml_toolkit.ml_utils import DSVUtils
from loguru import logger
from omegaconf import DictConfig

from kge.common.actions.kge_loader_action import KGEExperimentData
from kge.latency_profiler.profiler import KGELatencyProfiler


def run_latency_profiler(config: DictConfig, bundle: KGEExperimentData) -> None:
    model = bundle.model
    data_processor = bundle.data_processor
    profiler = KGELatencyProfiler(data_processor=data_processor, model=model)

    predict_df = profiler.run_predict(
        num_trials=config.task.num_trials,
        batch_size_list=config.task.batch_size_list,
        device_list=config.task.device_list,
    )

    score_df = profiler.run_score(
        num_trials=config.task.num_trials,
        batch_size_list=config.task.batch_size_list,
        device_list=config.task.device_list,
    )

    profiler_df = pd.concat([predict_df, score_df], ignore_index=True)

    file_path = Path(config.output_dir) / "time_profiler.tsv"

    DSVUtils.write_dsv(df=profiler_df, file_path=file_path, delimiter="\t")
    logger.info(f"Benchmark results saved to {file_path}")
