import sys

from dependency_injector.wiring import Provide, inject
from spaice_inference_api import ILogger, LoggerToken, Settings, create_server

from complex_query.container import QueryingContainer, QueryingSettings
from complex_query.delivery.rest.router import router


@inject
def on_startup_setup(logger: ILogger = Provide[LoggerToken]):
    logger.info("Initializing container")
    container = QueryingContainer()
    container.config.from_pydantic(QueryingSettings())

    container.wire(packages=["complex_query"])
    print(container.kg())


if __name__ == "__main__":
    server = create_server(
        model_loader=None,
        routers=[(None, router)],
        setup_ml_framework=on_startup_setup,
        wiring_config={"modules": [sys.modules[__name__]], "packages": ["complex_query"]},
        settings=Settings(),
    )
    server.run()
