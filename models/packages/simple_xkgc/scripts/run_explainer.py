from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
from dotenv import load_dotenv
from hakken_ml_toolkit.ml_utils import DSVUtils
from kge.common.actions.gnnkge_loader_action import GNNKGELoader
from kge.common.actions.kge_loader_action import KGELoader
from loguru import logger

from hakken_explainer.explainers import HakkenExplainer

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from hakken_explainer.candidate_finder.base import CandidateFinder
    from hakken_explainer.entities.config import ScoreTypeConfig
    from hakken_explainer.explainers import HakkenExplainerConfig


load_dotenv(override=False)


@hydra.main(
    version_base=None,
    config_path=os.getenv("CONFIG_PATH", "../config/"),
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    logger.remove()
    logger.add(sys.stderr, level=cfg.log_level)
    score_type_list: list[ScoreTypeConfig] = hydra.utils.instantiate(cfg.run.score_type_list)

    explainer_config: HakkenExplainerConfig = hydra.utils.instantiate(cfg.explainer)

    model_ckpt_folder = "seed_0/model_checkpoint"
    kge_folder = os.getenv("KGE_FOLDER")

    kge_bundle = KGELoader.load_experiment(
        experiment_folder=kge_folder,
        device=cfg.run.device,
        model_ckpt_path=os.path.join(model_ckpt_folder, "last.ckpt"),
        score_scaler_json_path=os.path.join(
            kge_folder, model_ckpt_folder, "last_score_scaler.json"
        ),
    )

    kge = kge_bundle.model

    del kge_bundle

    experiment_data = GNNKGELoader.load_experiment_from_config(
        config=explainer_config.gnn_experiment_config
    )
    model = experiment_data.model

    kg = experiment_data.data_processor.kg

    search_space = kg.get_encoded_facts(splits=explainer_config.search_space_split_names)

    candidate_finder: CandidateFinder = hydra.utils.instantiate(cfg.candidate_finder)
    candidate_finder.to_device(cfg.run.device)

    logger.info(f"[{cfg.run.device}]{candidate_finder}")
    logger.info(f"kge: {kge.device}")

    candidate_finder.setup(
        facts_batch=search_space, cache_folder=explainer_config.graph_cache_folder, kg=kg, kge=kge
    )

    explainer = HakkenExplainer(
        candidate_finder=candidate_finder, model=model, kg=kg, search_space=search_space
    )

    df_expl = explainer.explain(
        triple_to_probe=cfg.triple_to_probe,
        device=cfg.run.device,
        explanation_length=2,
        score_type_list=score_type_list,
        allowed_relations_ids=cfg.run.relation_filter,
    )

    df_expl.drop(columns=["explanation_index"], inplace=True)

    DSVUtils.write_dsv(
        df=df_expl,
        file_path=Path(cfg.output.path),
        delimiter=cfg.output.delimiter,
    )


if __name__ == "__main__":
    main()
