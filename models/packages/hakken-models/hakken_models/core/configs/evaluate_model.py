from pydantic import Field
from pydantic_settings import SettingsConfigDict

from .base_settings import HakkenSettings
from .evaluator import EvaluatorConfig
from .experiment_tracker import ExperimentTrackerConfig
from .model_loader import ModelLoaderConfig
from .train_common import DataLoaderConfig


class EvaluateModelConfig(HakkenSettings):
    """Configuration for model evaluation pipeline.

    This config is minimal and focused only on what's needed for evaluation:
    - Model source (MLflow run_id or directory path)
    - Evaluator configuration (metrics to compute)
    - Data loader configuration
    - Evaluation splits to run
    """

    model_loader: ModelLoaderConfig = Field(default_factory=ModelLoaderConfig)

    evaluator: EvaluatorConfig = Field(default_factory=EvaluatorConfig)

    experiment_tracker: ExperimentTrackerConfig = Field(default_factory=ExperimentTrackerConfig)

    data_loader: DataLoaderConfig = Field(default_factory=DataLoaderConfig)

    # Evaluation splits
    evaluation_groups: dict[str, list[str]] = Field(
        default_factory=lambda: {"train": ["train"], "val": ["train", "val"]},
        description="Dataset splits to evaluate, grouped by evaluation name",
    )

    model_config = SettingsConfigDict(env_prefix="EVALUATE_", case_sensitive=False)
