from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import pandas as pd
from dotenv import load_dotenv
from hakken_ml_toolkit.ml_base_structures import Triple
from hakken_ml_toolkit.ml_utils import DSVUtils
from hakken_ml_toolkit.ml_utils.function_timer import FunctionTimer, profile_method
from loguru import logger
from tqdm import tqdm

if TYPE_CHECKING:
    from hakken_explainer.path_finder.base import PathFinder
    from omegaconf import DictConfig

    from hakken_explainer.explainers.engine import PathExplainer


load_dotenv(override=False)


def setup_profiling(explainer: PathExplainer, timer: FunctionTimer) -> None:
    explainer.path_finder.find_paths = profile_method(timer, "find_paths")(
        explainer.path_finder.find_paths
    )

    explainer.explain = profile_method(timer, "explain")(explainer.explain)

    logger.info("Profiling enabled for explainer methods")


@hydra.main(
    version_base=None,
    config_path=os.getenv("CONFIG_PATH", "../config/"),
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    logger.remove()
    logger.add(sys.stderr, level=cfg.log_level)
    explanation_type_list = hydra.utils.instantiate(cfg.run.explanation_type_list)

    path_finder: PathFinder = hydra.utils.instantiate(cfg.path_finder)

    explainer: PathExplainer = hydra.utils.instantiate(cfg.explainer)
    explainer.setup(path_finder=path_finder)

    kg = explainer.kg
    entity_ids = kg.entity_mapping.get_ids()
    relation_ids = kg.relation_mapping.get_ids()
    data_list = []
    n_iterations = 200

    random.seed(42)

    timer = FunctionTimer()
    setup_profiling(explainer, timer)

    for _i in tqdm(range(n_iterations), desc="Benchmark", unit="iter"):
        sub = random.choice(entity_ids)
        rel = random.choice(relation_ids)
        obj = random.choice(entity_ids)

        triple = Triple(subject=sub, relation=rel, object=obj)
        k = explainer.explanation_len(triple)
        if k == -1:
            continue

        df_expl = explainer.explain(
            triple_to_probe=triple,
            device=cfg.run.device,
            explanation_type_list=explanation_type_list,
            allowed_relations_ids=None,
        )
        num_paths = df_expl.shape[0]

        time_stats = timer.get_stats()
        total_time = time_stats["explain"]["mean"]
        find_paths_time = time_stats["find_paths"]["mean"]

        data = {
            "triple": str(triple),
            "explanation_length": k,
            "total_seconds": total_time,
            "find_paths_seconds": find_paths_time,
            "num_paths": num_paths,
        }
        timer.reset()
        data_list.append(data)

    benchmark_df = pd.DataFrame(data_list)

    DSVUtils.write_dsv(
        df=benchmark_df,
        file_path=Path(cfg.output.path),
        delimiter=cfg.output.delimiter,
    )


if __name__ == "__main__":
    main()
