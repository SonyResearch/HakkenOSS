from __future__ import annotations

import os
import sys
from typing import cast

import hydra
import numpy as np
from dotenv import load_dotenv
from hakken_ml_toolkit.ml_utils.extras import PyTorchUtils
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from omegaconf import DictConfig

from kge.common.actions.kge_train import train_kge
from kge.utils import save_version

load_dotenv(override=False)


@hydra.main(
    version_base=None,
    config_path=os.getenv("CONFIG_PATH", "../../../config"),
    config_name="train_kge",
)
def main(cfg: DictConfig) -> float:
    logger.remove()
    logger.add(sys.stderr, level=cfg.log_level)

    PyTorchUtils.flush_gpu_memory()

    objective_list: list[float] = []
    hydra_cfg = HydraConfig.get()
    base_output_dir = hydra_cfg.runtime.output_dir

    save_version(base_output_dir)

    for seed in cfg.seed_list:
        cfg.run.output_dir = f"{base_output_dir}/seed_{seed}"

        objective = train_kge(cfg, seed=seed)
        objective_list.append(objective)

    objective = cast("float", np.mean(objective_list))

    logger.info(f"Experiment folder: {base_output_dir}")
    logger.info(f"Objective value: {objective:.4f}")
    return objective


if __name__ == "__main__":
    main()
