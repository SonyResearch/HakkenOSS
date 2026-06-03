from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import hydra
import pytorch_lightning as pl
import torch
from datasets import DataRepositoryI
from datasets.common.constants import DataSplits
from hakken_ml_toolkit.losses import RankingLossI
from hakken_ml_toolkit.ml_base_structures import KnowledgeGraph
from hakken_ml_toolkit.ml_utils import DictUtils
from hakken_ml_toolkit.ml_utils.extras import PyTorchUtils
from hakken_ml_toolkit.tracker import TrackerI
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from optuna import Trial
from optuna.integration import PyTorchLightningPruningCallback
from optuna.pruners import BasePruner
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from kge.common.entities import KGEDataBundle, TrainerCallbacks
from kge.initialization import BaseInitStrategy
from kge.models.base import KGEI
from kge.optim.factory import LRSchedulerInfo, OptimizerInfo
from kge.trainer.lightning import KGERankingLightning
from kge.trainer.utils import TrainUtils

if TYPE_CHECKING:
    from kge.data_processor import KGDataProcessor
    from kge.evaluator.base import KGEEvaluator
    from kge.negative_sampler.base import NegativeSamplerI


DataLoaderType = str
MetricType = str


def train_kge(cfg: DictConfig, seed: int | None = None) -> float:
    """
    Train a Knowledge Graph Embedding (KGE) model using PyTorch Lightning.

    This function orchestrates the complete training pipeline for KGE models, including
    data preparation, model initialization, training with optimal batch size discovery,
    evaluation, and artifact saving. It support experiment tracking.

    Args:
        cfg (DictConfig): Hydra configuration object containing all experiment parameters

    Returns:
        float: The value of the loss

    Side Effects:
        - Creates experiment output directory and saves configuration
        - Saves trained model, data repository, and data processor artifacts
        - Saves model embeddings to disk
        - Logs training progress and metrics to configured tracker
        - Sets PyTorch float32 matrix multiplication precision to "medium"
        - Fixes random seeds for reproducibility


    Notes:
        - Automatically discovers optimal batch size within specified constraints
        - Uses experiment tracking for logging configuration and metrics
        - Implements model checkpointing and early stopping
        - Saves best model when checkpointing is enabled
    """

    config: DictConfig = OmegaConf.create(cast("dict", OmegaConf.to_container(cfg, resolve=True)))

    torch.set_float32_matmul_precision(config.run.precision)

    if seed is not None:
        PyTorchUtils.fix_all_seeds(seed, device=config.run.device)

    output_dir = Path(cfg.run.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    data_bundle = KGETrainActions.prepare_data(config)

    tracker: TrackerI = hydra.utils.instantiate(config.tracker)

    model, loss_fn, negative_sampler = KGETrainActions.initialize_model_components(
        config=config, kg=data_bundle.kg, data_repo=data_bundle.data_repo
    )
    module, _evaluator, callbacks = KGETrainActions.setup_training(
        config=config,
        model=model,
        negative_sampler=negative_sampler,
        loss_fn=loss_fn,
        kg=data_bundle.kg,
        tracker=tracker,
    )

    train_dataset = data_bundle.data_processor.get_dataset(DataSplits.TRAIN)
    valid_dataset = data_bundle.data_processor.get_dataset(DataSplits.VALID)

    with tracker:
        tracker.track_config(
            DictUtils.flatten(cast("dict", OmegaConf.to_container(config, resolve=True)))
        )

        train_loader, valid_loader = TrainUtils.get_dataloaders(
            model_pl=module,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            batch_size_optimization_config=cfg.batch_size_optimization,
            loader_config=data_bundle.data_processor.loader_config,
            trainer_config=cfg.trainer,
        )

        tracker.track_config(
            DictUtils.flatten(cast("dict", OmegaConf.to_container(config, resolve=True)))
        )

        trainer = pl.Trainer(callbacks=callbacks.to_list(), **config.trainer)

        trainer.fit(model=module, train_dataloaders=train_loader, val_dataloaders=[valid_loader])

        model.to(config.run.device)

        objective = module.compute_objective_from_dataset(
            dataset=valid_dataset, device=config.run.device
        )
        tracker.track_value(key="validation/objective", value=objective)

    return objective


class KGETrainActions(Protocol):
    @staticmethod
    def prepare_data(
        config: DictConfig,
    ) -> KGEDataBundle:
        """
        Prepare data repository and processing pipeline.

        Args:
            config: The experiment configuration

        Returns:
            Tuple containing (data_repo, data_processor, train_loader, valid_loader)
        """
        # Initialize data repository
        data_repo: DataRepositoryI = hydra.utils.instantiate(config.data_repo)
        logger.info(f"Data Repo: {data_repo}")
        kg = data_repo.load_data()

        logger.info(f"Number of entities {kg.num_entities}")
        logger.info(f"Number of relations {kg.num_relations}")
        num_facts = kg.facts_dict["train"].shape[0]
        logger.info(f"Number of training_facts {num_facts}")

        # Initialize data processor
        data_processor: KGDataProcessor = hydra.utils.instantiate(config.data_processor)
        logger.info(f"Data Processor: {data_processor}")
        data_processor.set_kg(kg)

        return KGEDataBundle(data_repo, data_processor, kg)

    @staticmethod
    def initialize_model_components(
        config: DictConfig, kg: KnowledgeGraph, data_repo: DataRepositoryI
    ) -> tuple[KGEI, RankingLossI, NegativeSamplerI]:
        """
        Initialize model, loss function, and negative sampler.

        Args:
            config: The experiment configuration
            kg: Knowledge graph data
            data_repo: Data repository instance

        Returns:
            Tuple containing (model, loss_fn, negative_sampler)
        """
        # Initialize loss function
        loss_fn: RankingLossI = hydra.utils.instantiate(config.loss_fn)

        # Update model config with data statistics
        config.model.config.num_entities = data_repo.num_entities
        config.model.config.num_relations = data_repo.num_relations

        # Initialize and prepare model
        model: KGEI = hydra.utils.instantiate(config.model)
        init_strategy: BaseInitStrategy = hydra.utils.instantiate(config.init_strategy)

        init_strategy(model)

        # Initialize negative sampler
        negative_sampler: NegativeSamplerI = hydra.utils.instantiate(config.negative_sampler)
        negative_sampler.set_up(kg, config.run.device)

        return model, loss_fn, negative_sampler

    @staticmethod
    def get_callbacks(cfg: DictConfig) -> TrainerCallbacks:
        early_stopping: EarlyStopping = hydra.utils.instantiate(cfg.early_stopping)

        model_checkpoint: ModelCheckpoint | None = None
        if cfg.run.save_artifacts:
            model_checkpoint = hydra.utils.instantiate(cfg.model_checkpoint)

        trial: Trial = HydraConfig.get().runtime.get("trial")  # type: ignore[attr-defined]

        pruning_callback = None

        if trial is not None and isinstance(trial.study.pruner, BasePruner):
            monitor = "validation/loss"
            pruning_callback = PyTorchLightningPruningCallback(trial=trial, monitor=monitor)
        return TrainerCallbacks(
            early_stopping=early_stopping,
            model_checkpoint=model_checkpoint,
            pruning=pruning_callback,
        )

    @staticmethod
    def setup_training(
        config: DictConfig,
        model: KGEI,
        negative_sampler: NegativeSamplerI,
        loss_fn: RankingLossI,
        kg: KnowledgeGraph,
        tracker: TrackerI,
    ) -> tuple[KGERankingLightning, KGEEvaluator, TrainerCallbacks]:
        """
        Set up training components including optimizer, scheduler, and evaluator.

        Args:
            config: The experiment configuration
            model: The KGE model
            negative_sampler: Negative sampler instance
            loss_fn: Loss function
            kg: Knowledge graph data

        Returns:
            Tuple containing (lightning_module, callbacks)
        """
        # Initialize optimizer and learning rate scheduler
        optimizer_info: OptimizerInfo = hydra.utils.instantiate(config.optimizer)
        lr_sched_info: LRSchedulerInfo = hydra.utils.instantiate(config.lr_scheduler)

        # Initialize evaluator
        evaluator: KGEEvaluator = hydra.utils.instantiate(config.evaluator)
        evaluator.init(kg)

        # Create lightning module
        module = KGERankingLightning(
            model=model,
            negative_sampler=negative_sampler,
            optimizer_info=optimizer_info,
            lr_sched_info=lr_sched_info,
            loss_fn=loss_fn,
            evaluator=evaluator,
            tracker=tracker,
            remove_triples_path=config.run.remove_triples_path,
        )

        callbacks = KGETrainActions.get_callbacks(config)

        return module, evaluator, callbacks
