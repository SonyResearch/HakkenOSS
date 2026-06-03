import sys

from complex_query.container import QueryingContainer, QueryingSettings
from complex_query.delivery.rest.router import router as complex_router
from dependency_injector.wiring import Provide, inject
from simple_query.api.container import ApiConfig, SimpleQueryingContainer
from simple_query.api.router import router as simple_router
from spaice_inference_api import ILogger, LoggerToken, Settings, create_server


@inject
def on_startup_setup(logger: ILogger = Provide[LoggerToken]):
    logger.info("Initializing container")
    container = QueryingContainer()
    container.config.from_pydantic(QueryingSettings())

    container.wire(packages=["complex_query"])
    logger.info("Initializing simple query container")
    simple_container = SimpleQueryingContainer()
    simple_container.config.from_pydantic(ApiConfig())  # type: ignore
    simple_container.wire(packages=["simple_query"])


if __name__ == "__main__":
    server = create_server(
        model_loader=None,
        routers=[("/simple", simple_router), ("/complex", complex_router)],
        setup_ml_framework=on_startup_setup,
        wiring_config={
            "modules": [sys.modules[__name__]],
            "packages": ["complex_query", "simple_query"]
        },
        settings=Settings(),
    )
    server.run()
