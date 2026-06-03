from time import time

from dependency_injector.wiring import Provide, inject

from spaice_inference_api.core.contract.logger import ILogger, LoggerToken
from spaice_inference_api.core.contract.model import (
    IModel,
    IModelLoader,
    ModelLoaderToken,
    ModelLoadingOptions,
)
from spaice_inference_api.core.errors import ModelLoadingError


class LoadModelAction:
    @inject
    def go(
        self,
        options: ModelLoadingOptions,
        logger: ILogger = Provide[LoggerToken],
        model_loader: IModelLoader = Provide[ModelLoaderToken],
    ) -> IModel:
        try:
            logger.info(f'Loading model from path "{options.path}"')
            loading_start = time()
            model = model_loader.load(options)
            logger.info(f"Model was loaded successfully in {time() - loading_start} seconds")
            return model
        except Exception as e:
            logger.exception(e)
            raise ModelLoadingError(f'Loading model failed with error: "{e!s}"') from e
