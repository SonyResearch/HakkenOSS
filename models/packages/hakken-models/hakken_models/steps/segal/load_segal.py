"""Step to load trained SeGAL artifacts."""

from typing import Annotated

from loguru import logger
from zenml import ArtifactConfig, step
from zenml.client import Client

from hakken_models.core.configs.model_loader import ModelLoaderConfig
from hakken_models.datasets.deployment import DatasetDeployment
from hakken_models.models.segal import SeGAL, SeGALLoader

experiment_tracker = Client().active_stack.experiment_tracker


@step(
    enable_cache=False,
    experiment_tracker=experiment_tracker.name,
)
def load_segal_artifacts_step(
    config: ModelLoaderConfig,
) -> tuple[
    Annotated[SeGAL, ArtifactConfig(name="segal")],
    Annotated[DatasetDeployment | None, ArtifactConfig(name="dataset")],
]:
    """Load a trained SeGAL model from MLflow or directory.

    Args:
        config: Model loader configuration containing model source information.

    Returns:
        Tuple of (loaded SeGAL model, dataset deployment).
    """
    loader = SeGALLoader(config)
    artifacts = loader.load()

    logger.success("SeGAL artifacts loaded successfully.")

    return artifacts.segal, artifacts.dataset
