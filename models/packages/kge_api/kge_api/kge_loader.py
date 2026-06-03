from __future__ import annotations

import os
from typing import TYPE_CHECKING

from datasets.common.constants import DataSplits
from dependency_injector.wiring import Provide, inject
from kge.common.actions.kge_loader_action import KGEExperimentData, KGELoader
from spaice_inference_api import ILogger, IModelLoader, LoggerToken, ModelLoadingOptions

from kge_api.config import APIConfig
from kge_api.container import Container

if TYPE_CHECKING:
    from kge_api.config import APIConfig


def prepare_score_scaler(
    kge_bundle: KGEExperimentData,
    score_scaler_json_path: str | None,
    overwrite: bool,
    device: str,
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


class KGERunLoader(IModelLoader):
    @inject
    def load(
        self,
        _options: ModelLoadingOptions,
        logger: ILogger = Provide[LoggerToken],
        config: APIConfig = Provide[Container.config],
    ) -> KGEExperimentData:
        kge_bundle = KGELoader.load_experiment(
            experiment_folder=config.experiment_folder,
            config_path=config.config_path,
            model_ckpt_path=config.model_ckpt_path,
            model_ckpt_is_lightning=config.model_ckpt_is_lightning,
            device=config.device,
            load_negative_sampler=True,
        )

        if config.score_scaler_json_path is not None:
            logger.info(f"Loading score scaler from {config.score_scaler_json_path}...")
            success = kge_bundle.model.load_score_scaler(config.score_scaler_json_path)
            if not success:
                prepare_score_scaler(
                    kge_bundle=kge_bundle,
                    score_scaler_json_path=config.score_scaler_json_path,
                    overwrite=True,
                    device=config.device,
                    loader_kwargs=None,
                )
            logger.info("Score scaler loaded!")
        return kge_bundle
