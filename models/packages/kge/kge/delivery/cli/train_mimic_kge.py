from __future__ import annotations

import os
import sys
from typing import cast

import hydra
import numpy as np
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from omegaconf import DictConfig

from kge.trainer.mimic_kge.action import train_mimic_kge

load_dotenv()


@hydra.main(
    version_base=None,
    config_path=os.getenv("CONFIG_PATH", "../../../config"),
    config_name="train_mimic_kge",
)
def main(cfg: DictConfig) -> float:
    logger.remove()
    logger.add(sys.stderr, level=cfg.log_level)

    objective_list: list[float] = []
    hydra_cfg = HydraConfig.get()
    base_output_dir = hydra_cfg.runtime.output_dir
    for seed in cfg.seed_list:
        cfg.run.output_dir = f"{base_output_dir}/seed_{seed}"

        objective_i = train_mimic_kge(cfg, seed=seed)
        objective_list.append(objective_i)

    objective = cast("float", np.mean(objective_list))
    logger.info(f"Experiment folder: {base_output_dir}")
    logger.info(f"Objective value: {objective:.4f}")
    return objective


if __name__ == "__main__":
    main()
