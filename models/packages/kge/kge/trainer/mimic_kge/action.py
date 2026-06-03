from pathlib import Path
from typing import cast

import hydra
from datasets.common.constants import DataSplits
from hakken_ml_toolkit.ml_utils import DictUtils
from hakken_ml_toolkit.ml_utils.extras import PyTorchUtils
from hakken_ml_toolkit.tracker import TrackerI
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

from kge.common.actions.kge_train import KGETrainActions
from kge.initialization import BaseInitStrategy
from kge.models.gnn import GNNKGE, GNNKGEConfig
from kge.models.kge_api import KGEAPI
from kge.optim.factory import LRSchedulerInfo, OptimizerInfo
from kge.trainer.mimic_kge import MimicKGELightning


def train_mimic_kge(cfg: DictConfig, seed: int | None = None) -> float:
    config: DictConfig = OmegaConf.create(cast("dict", OmegaConf.to_container(cfg, resolve=True)))

    if seed is not None:
        PyTorchUtils.fix_all_seeds(seed, device=config.run.device)

    output_dir = Path(cfg.run.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    data_bundle = KGETrainActions.prepare_data(config)

    data_processor = data_bundle.data_processor

    trained_kge: KGEAPI = hydra.utils.instantiate(config.trained_kge)

    train_loader = data_processor.get_mimic_kge_data_loader(
        split=DataSplits.TRAIN, trained_kge=trained_kge
    )
    train_loader.fit_scaler()

    valid_loader = data_processor.get_mimic_kge_data_loader(
        split=DataSplits.VALID,
        trained_kge=trained_kge,
        subgraph_split=DataSplits.TRAIN,
        shuffle=False,
    )
    valid_loader.load_scaler()

    model_config: GNNKGEConfig = hydra.utils.instantiate(config.model)

    model = GNNKGE.from_config(config=model_config, dataset=data_bundle.data_repo).to(
        cfg.run.device
    )

    init_strategy: BaseInitStrategy = hydra.utils.instantiate(config.init_strategy)

    init_strategy(model)

    tracker: TrackerI = hydra.utils.instantiate(config.tracker)

    optimizer_info: OptimizerInfo = hydra.utils.instantiate(config.optimizer)

    lr_sched_info: LRSchedulerInfo = hydra.utils.instantiate(config.lr_scheduler)

    with tracker:
        tracker.track_config(
            DictUtils.flatten(cast("dict", OmegaConf.to_container(config, resolve=True)))
        )

        lit_model = MimicKGELightning(
            optimizer_info=optimizer_info,
            lr_sched_info=lr_sched_info,
            model=model,
            tracker=tracker,
        )
        callbacks = KGETrainActions.get_callbacks(config)

        trainer = Trainer(callbacks=callbacks.to_list(), **config.trainer)

        trainer.fit(
            model=lit_model,
            train_dataloaders=train_loader,
            val_dataloaders=[valid_loader],
        )

        objective = lit_model.compute_objective_from_dataloader(
            dataloader=valid_loader, device=config.run.device
        )
        tracker.track_value(key="validation/objective", value=objective)

    return objective
