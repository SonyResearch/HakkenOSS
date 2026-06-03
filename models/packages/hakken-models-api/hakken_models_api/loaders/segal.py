"""SeGAL model loader."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dependency_injector.wiring import Provide, inject
from hakken_models.models.segal import SeGALArtifacts, SeGALLoader
from spaice_inference_api import ILogger, IModelLoader, LoggerToken, ModelLoadingOptions

from hakken_models_api.container import Container

if TYPE_CHECKING:
    from hakken_models_api.config import HakkenModelsAPIConfig


class SeGALRunLoader(IModelLoader):
    """IModelLoader implementation that loads SeGAL from MLflow or directory."""

    @inject
    def load(
        self,
        options: ModelLoadingOptions,
        logger: ILogger = Provide[LoggerToken],
        config: HakkenModelsAPIConfig = Provide[Container.config],
    ) -> SeGALArtifacts:
        logger.debug(f"SeGALRunLoader.load called with options: {options}")
        source = config.mlflow_run_id or config.run_dir
        logger.info(f"Loading SeGAL from {source}...")

        loader = SeGALLoader(config)
        artifacts = loader.load()

        logger.info("SeGAL loaded successfully.")
        return artifacts
