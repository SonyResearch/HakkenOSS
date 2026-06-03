from typing import Annotated

from loguru import logger
from zenml import ArtifactConfig, step
from zenml.client import Client

from hakken_models.core.configs.model_loader import ModelLoaderConfig
from hakken_models.datasets.deployment import DatasetDeployment
from hakken_models.models.thiger import THiGER, THiGERLoader

experiment_tracker = Client().active_stack.experiment_tracker


@step(
    enable_cache=False,
    experiment_tracker=experiment_tracker.name,
)
def load_thiger_artifacts_step(
    config: ModelLoaderConfig,
) -> tuple[
    Annotated[THiGER, ArtifactConfig(name="thiger")],
    Annotated[DatasetDeployment | None, ArtifactConfig(name="dataset")],
]:
    """Load a trained THiGER model from MLflow or directory.

    Args:
        config: Evaluation configuration containing model source information.

    Returns:
        Tuple of (loaded THiGER model, dataset metadata dict, dataset config).
    """

    loader = THiGERLoader(config)

    artifacts = loader.load()

    logger.success("THiGER artifacts loaded successfully.")

    return artifacts.thiger, artifacts.dataset
