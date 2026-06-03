"""THiGER model loader."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dependency_injector.wiring import Provide, inject
from spaice_inference_api import ILogger, IModelLoader, LoggerToken, ModelLoadingOptions

from hakken_models_api.container import Container

if TYPE_CHECKING:
    from hakken_models.models.thiger import THiGERArtifacts

    from hakken_models_api.config import HakkenModelsAPIConfig


class THiGERRunLoader(IModelLoader):
    @inject
    def load(
        self,
        options: ModelLoadingOptions,
        logger: ILogger = Provide[LoggerToken],
        config: HakkenModelsAPIConfig = Provide[Container.config],
    ) -> THiGERArtifacts:
        logger.debug(f"THiGERRunLoader.load called with options: {options}")
        logger.info(f"Loading THiGER from run_id {config.mlflow_run_id}...")
        raise NotImplementedError("THiGERRunLoader is not implemented.")
