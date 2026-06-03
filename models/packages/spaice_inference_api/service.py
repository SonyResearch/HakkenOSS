import sys

from dependency_injector.wiring import Provide, inject

from service_sample.model import MyModel
from service_sample.router import router
from spaice_inference_api import (
    ILogger,
    IModelLoader,
    LoggerToken,
    ModelLoadingOptions,
    create_server,
)


class MyModelLoader(IModelLoader):
    @inject
    def load(self, options: ModelLoadingOptions, logger: ILogger = Provide[LoggerToken]) -> MyModel:
        # Specifies the model loader class responsible for loading ML models.
        # Called automatically you just have to set it and pass it to the create_server method
        logger.info(f"Will load the model from options path: '{options.path}'")
        return MyModel()


@inject
def dummy_framework_setup(logger: ILogger = Provide[LoggerToken]):
    # You can use this to do heavy things on startup instead of runtime
    logger.info("This is not loading anything, it's dummy")


# EXAMPLE: If you have a model to load you can do the following
server = create_server(
    model_loader=MyModelLoader,  # Specifies the loader class responsible for loading ML models.
    routers=[
        router
    ],  # Adds the defined API router(s) for handling HTTP routes. eg your prediction router
    setup_ml_framework=dummy_framework_setup,  # Function to set up the ML framework
    # (e.g., initializing TensorFlow or PyTorch).
    wiring_config={  # Dependency injection wiring configuration for the inference_api container
        "modules": [sys.modules[__name__]],  # Modules to inspect for dependency injection setup.
        "packages": ["service_sample"],  # Additional packages to scan for services or dependencies.
    },
)


server.run()
