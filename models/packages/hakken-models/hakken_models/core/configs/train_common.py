from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Self

import mlflow
from loguru import logger
from omegaconf import OmegaConf
from pydantic import BaseModel, Field
from pydantic_settings import SettingsConfigDict

from hakken_models.core.utils.data import unflatten_dict
from hakken_models.evaluators.metric_hub import MetricHubConfig

from .base_settings import HakkenSettings
from .early_stopping import EarlyStoppingConfig
from .experiment_tracker import ExperimentTrackerConfig
from .hpo import HPOConfig
from .negative_strategy import NegativeStrategyConfig


class DatasetConfig(BaseModel):
    name: str = "pubtator3-v0.4.0"
    version: str = "v1"
    data_root_uri_template: str = Field(
        default="s3://sai-spaice-ds/data/processed/data_processing/zenml/{name}/{version}",
        description="Template for dataset root URI. Supports variables {name} and {version}.",
    )
    load_embeddings: bool = Field(
        default=True,
        description="Whether to load pre-computed node/relation embeddings. Set to False for KGE training.",
    )

    @property
    def data_root_uri(self) -> str:
        return self.data_root_uri_template.format(name=self.name, version=self.version)


class DataLoaderConfig(BaseModel):
    name: str = "DataLoader"
    kwargs: dict[str, Any] = Field(
        default_factory=lambda: {"batch_size": 1024, "num_workers": 4, "pin_memory": True}
    )


class InitStrategyConfig(BaseModel):
    name: str = "XavierNormal"
    kwargs: dict[str, Any] = Field(default_factory=lambda: {"gain": 1.0, "skip_bias_init": False})


class LoggerConfig(BaseModel):
    log_model: Literal[True, False, "all"] = False


class LossConfig(BaseModel):
    name: str = "MarginRankingLoss"
    kwargs: dict[str, Any] = Field(default_factory=lambda: {"margin": 1.0})

    def with_kge_negative_strategy(
        self, negative_strategy: NegativeStrategyConfig
    ) -> dict[str, Any]:
        """Hydra-style loss dict for KGE with ``neg_strategy`` from ``negative_strategy``.

        ``negative_strategy.name`` is the single source of truth for hardest/mean
        aggregation; any ``neg_strategy`` key under ``kwargs`` is overwritten so
        training cannot diverge from the logged train config.
        """
        out = self.model_dump()
        kwargs = dict(out.get("kwargs", {}))
        kwargs["neg_strategy"] = negative_strategy.name.value
        out["kwargs"] = kwargs
        return out


class ModelCheckpointConfig(BaseModel):
    checkpoint_dir: str = "checkpoints"
    filename: str = "best-{epoch:02d}-{val_loss:.2f}"
    monitor: str = "val_loss"
    mode: str = "min"
    save_top_k: int = 1
    save_last: bool = True
    every_n_epochs: int = 1
    enabled: bool = True


class NegSamplerConfig(BaseModel):
    name: str = "UniformNegativeSampler"
    kwargs: dict[str, Any] = Field(
        default_factory=lambda: {"corruption_scheme": ["subject", "object"]}
    )


class OptimizerConfig(BaseModel):
    name: str = "Adam"
    kwargs: dict[str, Any] = Field(
        default_factory=lambda: {"lr": 0.001, "betas": [0.9, 0.999], "weight_decay": 0.0}
    )


class ResumeCheckpointConfig(BaseModel):
    uri: str
    local_dir: str = "checkpoints"

    @property
    def uri_type(self) -> str:
        if self.uri.startswith("mlflow:/") or self.uri.startswith("runs:/"):
            return "mlflow"
        return "local"


class RunConfig(BaseModel):
    """Configuration for experiment management."""

    seed: int | None = 42
    cleanup_local_checkpoints: bool = True


class SchedulerConfig(BaseModel):
    name: str = "ReduceLROnPlateau"
    kwargs: dict[str, Any] = Field(
        default_factory=lambda: {"mode": "min", "factor": 0.1, "patience": 10}
    )


class TrainerConfig(BaseModel):
    limit_train_batches: int | float | None = None
    limit_val_batches: int | float | None = None
    max_epochs: int = 2
    devices: int = 1
    strategy: str = "auto"
    precision: str | None = None
    check_val_every_n_epoch: int = 1
    auto_batch_size: bool = Field(
        default=False,
        description=(
            "Not forwarded to Lightning Trainer. KGE and SeGAL training steps always "
            "run find_max_batch_size before fit."
        ),
    )
    gradient_clip_val: float | None = Field(
        default=1.0,
        description="Gradient clipping by value. None disables. Helps prevent NaN from exploding gradients.",
    )


class BaseTrainConfig(HakkenSettings):
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    data_loader: DataLoaderConfig = Field(default_factory=DataLoaderConfig)
    init_strategy: InitStrategyConfig = Field(default_factory=InitStrategyConfig)
    logger: LoggerConfig = Field(default_factory=LoggerConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    model_checkpoint: ModelCheckpointConfig = Field(default_factory=ModelCheckpointConfig)
    resume: ResumeCheckpointConfig | None = Field(default=None)
    early_stopping: EarlyStoppingConfig | None = Field(default=None)
    experiment_tracker: ExperimentTrackerConfig = Field(default_factory=ExperimentTrackerConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    run: RunConfig = Field(default_factory=RunConfig)
    scheduler: SchedulerConfig | None = None
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    hpo: HPOConfig | None = None
    val_metric_hub: MetricHubConfig = Field(default_factory=MetricHubConfig)

    model_config = SettingsConfigDict(
        env_prefix="BASE_TRAIN_",
        case_sensitive=False,
        extra="ignore",
    )

    @classmethod
    def from_file(
        cls,
        path: Path | str,
        *,
        overrides: list[str] | None = None,
    ) -> Self:
        """Build config from a nested YAML file (merged tree, not Hydra defaults).

        The YAML shape must match the concrete subclass (e.g. ``TrainKGEConfig``).
        Optional ``overrides`` use Hydra-style dotted keys (``key=value``, e.g.
        ``run.seed=99``), merged on top with ``OmegaConf.merge`` after load.

        Args:
            path: Path to the YAML file.
            overrides: Dotted-key overrides merged after load.

        Returns:
            Parsed instance of this class.

        Raises:
            FileNotFoundError: If ``path`` is missing or not a regular file.
        """
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            msg = f"Config YAML not found: {resolved}"
            raise FileNotFoundError(msg)

        cfg = OmegaConf.load(resolved)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
        OmegaConf.resolve(cfg)
        return cls.from_omegaconf(cfg)

    @classmethod
    def from_mlflow(
        cls,
        run_id: str,
        *,
        tracking_uri: str | None = None,
        param_overrides: dict[str, str] | None = None,
    ) -> Self:
        """Build config from an MLflow run's logged parameters.

        Expects the flat ``/``-separated key layout produced by
        ``log_hyperparameters_step``. Optional ``param_overrides`` merge on top
        (same flat keys, e.g. ``{"run/seed": "123"}``).

        Concrete configs (:class:`~hakken_models.core.configs.train_kge.TrainKGEConfig`,
        :class:`~hakken_models.core.configs.train_segal.TrainSeGALConfig`, etc.) inherit
        this implementation.

        Args:
            run_id: MLflow run ID.
            tracking_uri: If set, passed to :func:`mlflow.set_tracking_uri` before lookup.
            param_overrides: Flat string overrides merged into run params.

        Returns:
            Parsed training configuration instance.

        Raises:
            RuntimeError: If the run has no parameters.
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        run = mlflow.get_run(run_id=run_id)
        logger.info(f"Building {cls.__name__} from MLflow run_id {run_id}")

        flat_params = dict(run.data.params)
        if len(flat_params) == 0:
            raise RuntimeError("No params in run (unexpected).")

        flat_params.pop("last_checkpoint_path", None)
        flat_params.pop("checkpoint_dir", None)

        if param_overrides:
            logger.info(f"Applying param overrides: {param_overrides}")
            flat_params.update(param_overrides)

        nested = unflatten_dict(flat_params, sep="/")
        cfg = OmegaConf.create(nested)
        return cls.from_omegaconf(cfg)
