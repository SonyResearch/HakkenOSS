import sys

from dependency_injector.wiring import Provide, inject
from spaice_inference_api import ILogger, LoggerToken, create_server
from spaice_inference_api import Settings as InferenceSettings

from contextualization.api.container import ApiConfig, ContextualizationContainer
from contextualization.api.router import router


@inject
def on_startup_setup(logger: ILogger = Provide[LoggerToken]) -> None:
    logger.info("Initializing container")
    container = ContextualizationContainer()
    container.config.from_pydantic(ApiConfig())  # type: ignore
    container.wire(packages=["contextualization"])


if __name__ == "__main__":
    server = create_server(
        model_loader=None,
        routers=[(None, router)],
        setup_ml_framework=on_startup_setup,
        wiring_config={"modules": [sys.modules[__name__]], "packages": ["contextualization"]},
        settings=InferenceSettings(),
    )
    server.run()
