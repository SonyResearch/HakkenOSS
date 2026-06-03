from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import hydra
import torch
from hakken_ml_toolkit.ml_utils import YAMLUtils

if TYPE_CHECKING:
    from kge.models.base import KGEI

import os
from dataclasses import dataclass

from datasets.common.constants import DataSplits
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from kge.common.actions.kge_train import KGETrainActions
from kge.common.entities.kge_loader_config import KGELoadExperimentConfig
from kge.data_processor import KGDataProcessor
from kge.negative_sampler import NegativeSamplerI


@dataclass
class KGEExperimentData:
    model: KGEI
    data_processor: KGDataProcessor
    negative_sampler: NegativeSamplerI | None = None


def prepare_score_scaler(
    kge_bundle: KGEExperimentData,
    score_scaler_json_path: str | None,
    overwrite: bool,
    device: str | torch.device,
    loader_kwargs: dict | None = None,
) -> None:
    model = kge_bundle.model

    data_processor = kge_bundle.data_processor
    negative_sampler = kge_bundle.negative_sampler

    if negative_sampler is not None:
        train_dataset = data_processor.get_dataset(split=DataSplits.TRAIN, num_triples=1024 * 20)

        model.to_device(device)

        if (
            overwrite
            and score_scaler_json_path is not None
            and os.path.exists(score_scaler_json_path)
        ):
            os.remove(score_scaler_json_path)

        model.fit_score_scaler_from_dataset(
            dataset=train_dataset,
            negative_sampler=kge_bundle.negative_sampler,
            json_path=score_scaler_json_path,
            loader_kwargs=loader_kwargs,
        )


class KGELoader(Protocol):
    @staticmethod
    def load_experiment_from_config(
        config: KGELoadExperimentConfig,
    ) -> KGEExperimentData:
        return KGELoader.load_experiment(**config.model_dump())

    @staticmethod
    def load_experiment(
        experiment_folder: str | Path,
        config_path: str = ".hydra/config.yaml",
        model_ckpt_path: str = "seed_0/model_checkpoint/last.ckpt",
        model_ckpt_is_lightning: bool = True,
        device: str | torch.device = "cpu",
        load_negative_sampler: bool = False,
        score_scaler_json_path: str | None = None,
    ) -> KGEExperimentData:
        if isinstance(experiment_folder, str):
            experiment_folder = Path(experiment_folder)

        config_file = experiment_folder / config_path
        cfg = cast("DictConfig", OmegaConf.load(config_file))

        data_bundle = KGETrainActions.prepare_data(cfg)

        data_repo = data_bundle.data_repo

        kg = data_repo.load_data()

        data_processor = data_bundle.data_processor

        data_processor.set_kg(kg)

        cfg.model.config.num_entities = data_repo.num_entities
        cfg.model.config.num_relations = data_repo.num_relations

        model: KGEI = hydra.utils.instantiate(cfg.model)

        checkpoint = torch.load(experiment_folder / model_ckpt_path, weights_only=False)
        if model_ckpt_is_lightning:
            state_dict = {key[6:]: value for key, value in checkpoint["state_dict"].items()}
        else:
            state_dict = checkpoint["state_dict"]
        model.load_state_dict(state_dict)
        model.to(device)

        data_processor.to(device)

        negative_sampler: NegativeSamplerI | None = None
        if load_negative_sampler:
            negative_sampler = cast(
                "NegativeSamplerI", hydra.utils.instantiate(cfg.negative_sampler)
            )
            negative_sampler.set_up(kg=kg, device=device)

        kge_bundle = KGEExperimentData(
            model=model,
            data_processor=data_processor,
            negative_sampler=negative_sampler,
        )

        if score_scaler_json_path is not None:
            logger.info(f"Loading score scaler from {score_scaler_json_path}...")
            success = kge_bundle.model.load_score_scaler(score_scaler_json_path)
            if not success:
                prepare_score_scaler(
                    kge_bundle=kge_bundle,
                    score_scaler_json_path=score_scaler_json_path,
                    overwrite=True,
                    device=device,
                    loader_kwargs=None,
                )
            logger.info("Score scaler loaded!")
        return kge_bundle

    @staticmethod
    def load(
        experiment_path: str | Path,
        ckpt_folder: str = "kge",
        device: str | torch.device = "cpu",
    ) -> KGEI:
        model_path = Path(experiment_path) / ckpt_folder

        config_train_path = Path(experiment_path) / ".hydra" / "config.yaml"
        config_train = YAMLUtils.load(config_train_path)

        kge_class_str = config_train["model"]["_target_"]

        kge_class: type[KGEI] = hydra.utils.get_class(kge_class_str)

        return kge_class.load(model_path, device=device)
