from __future__ import annotations

import os
import sys

import hydra
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from kge.common.actions.gnnkge_loader_action import GNNKGELoader
from kge.common.actions.kge_loader_action import KGELoader
from kge.common.inference_tasks import INFERENCE_TASK_MAP
from kge.utils import load_version


@hydra.main(
    version_base=None,
    config_path=os.getenv("CONFIG_PATH", "../../../config"),
    config_name="inference",
)
def main(cfg: DictConfig) -> None:
    logger.remove()
    logger.add(sys.stderr, level=cfg.log_level)

    success = load_version(cfg.experiment_folder)
    if not success:
        msg = "Could not load version information from experiment folder."
        logger.warning(msg)

    load_dotenv(override=False)

    config = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    loader_class: KGELoader | GNNKGELoader = KGELoader

    loader_kwargs = {
        "experiment_folder": config.experiment_folder,
        "config_path": config.config_path,
        "model_ckpt_path": config.model_ckpt_path,
        "model_ckpt_is_lightning": config.model_ckpt_is_lightning,
        "score_scaler_json_path": config.score_scaler_json_path,
        "device": config.device,
    }

    if config.model_is_gnn:
        loader_class = GNNKGELoader
        loader_kwargs["load_trained_kge"] = True

    bundle = loader_class.load_experiment(**loader_kwargs)

    task_fn = INFERENCE_TASK_MAP.get(config.task.name)

    task_fn(config, bundle)


if __name__ == "__main__":
    main()
