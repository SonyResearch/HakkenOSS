from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from kge.common.actions.kge_train import KGETrainActions
from kge.common.entities.kge_loader_config import KGELoadExperimentConfig
from kge.data_processor import KGDataProcessor
from kge.models.gnn import GNNKGE, GNNKGEConfig
from kge.models.kge_api import KGEAPI


@dataclass
class GNNKGEExperimentData:
    model: GNNKGE
    data_processor: KGDataProcessor
    trained_kge: KGEAPI | None = None


class GNNKGELoader(Protocol):
    @staticmethod
    def load_experiment_from_config(
        config: KGELoadExperimentConfig,
    ) -> GNNKGEExperimentData:
        return GNNKGELoader.load_experiment(**config.model_dump())

    @staticmethod
    def load_experiment(
        experiment_folder: str | Path,
        config_path: str = ".hydra/config.yaml",
        model_ckpt_path: str = "seed_0/model_checkpoint/last.ckpt",
        model_ckpt_is_lightning: bool = True,
        device: str | torch.device = "cpu",
        load_trained_kge: bool = False,
    ) -> GNNKGEExperimentData:
        if isinstance(experiment_folder, str):
            experiment_folder = Path(experiment_folder)

        config_file = experiment_folder / config_path
        cfg = cast("DictConfig", OmegaConf.load(config_file))

        data_bundle = KGETrainActions.prepare_data(cfg)

        data_repo = data_bundle.data_repo

        kg = data_repo.load_data()

        data_processor = data_bundle.data_processor

        data_processor.set_kg(kg)

        model_config: GNNKGEConfig = hydra.utils.instantiate(cfg.model)

        model = GNNKGE.from_config(config=model_config, dataset=data_bundle.data_repo).to(device)

        checkpoint = torch.load(experiment_folder / model_ckpt_path, weights_only=False)
        if model_ckpt_is_lightning:
            state_dict = {key[6:]: value for key, value in checkpoint["state_dict"].items()}
        else:
            state_dict = checkpoint["state_dict"]
        model.load_state_dict(state_dict)
        model.to(device)

        data_processor.to(device)

        trained_kge: KGEAPI | None = None
        if load_trained_kge:
            trained_kge = cast("KGEAPI", hydra.utils.instantiate(cfg.trained_kge))

        return GNNKGEExperimentData(
            model=model, data_processor=data_processor, trained_kge=trained_kge
        )
