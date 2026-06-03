import logging
from functools import partial
from typing import TYPE_CHECKING

import uvicorn
from dependency_injector import providers
from fastapi import APIRouter, FastAPI
from typing_extensions import TypedDict

from spaice_inference_api.config import Settings
from spaice_inference_api.container import Container
from spaice_inference_api.delivery.rest.infra_controller import (
    router as infra_controller_router,
)
from spaice_inference_api.utils.app.events import on_shutdown, on_startup
from spaice_inference_api.utils.app.middlewares import (
    authentication_middleware,
    request_id_middleware,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from spaice_inference_api.core.contract.logger import ILogger
    from spaice_inference_api.core.contract.model import IModelLoader


def create_app(model_name: str, routers: list[tuple[str | None, APIRouter]]) -> FastAPI:
    app = FastAPI()
    app.include_router(infra_controller_router)
    app.include_router(infra_controller_router, prefix=f"/{model_name}")
    for extra_prefix, router in routers:
        full_prefix = f"/{model_name}{extra_prefix}" if extra_prefix else f"/{model_name}"
        app.include_router(router, prefix=full_prefix)
    app.middleware("http")(authentication_middleware)
    app.middleware("http")(request_id_middleware)

    return app


class WiringConfig(TypedDict, total=False):
    packages: "Iterable"
    modules: "Iterable"


def create_server(
    model_loader: type["IModelLoader"] | None,
    setup_ml_framework: "Callable | None",
    routers: list[tuple[str | None, APIRouter]] | None = None,
    wiring_config: WiringConfig | None = None,
    settings: Settings | None = None,
) -> uvicorn.Server:
    """
    Creates and configures a `uvicorn.Server` instance for hosting an API.
    This function integrates model loading, framework setup, routing, and
    server configuration.

    Args:
        model_loader (Type[IModelLoader] | None):
            An implementation of the `IModelLoader` interface responsible
            for loading and managing machine learning models. If `None`,
            no model loading occurs.
        setup_ml_framework (Callable | None):
            A callable for setting up the machine learning framework
            (e.g., TensorFlow, PyTorch). If `None`, this step is skipped.
            You can use this also for anything you would like to run
            durint the startup of your server, like loading anything
            to save time later.
        routers (List[APIRouter], optional):
            A list of FastAPI `APIRouter` objects defining API endpoints
            to include in the server. Defaults to an empty list.
        wiring_config (WiringConfig, optional):
            Configuration for dependency injection wiring. Defaults to an
            empty dictionary. Include the modules and packages you worked
            on to make sure the inference_api container is injected.
        settings (Settings, optional):
            A `Settings` instance containing server configurations such as
            host, port, and logging. Defaults to a new `Settings` instance.
            Create and extend the Settings class to create your own settings

    Returns:
        uvicorn.Server:
            A configured `uvicorn.Server` instance, ready to start the
            application.

    """
    if settings is None:
        settings = Settings()
    if wiring_config is None:
        wiring_config = {}
    if routers is None:
        routers = []
    app = create_app(settings.SPAICE_MODEL_NAME, routers)

    app_container = Container()

    app_container.config.from_dict(
        {
            "infra_utils": {
                "logger_name": "spaice_inference_api",
            }
        }
    )

    app_container.init_resources()

    if model_loader is not None:
        app_container.model_loader.override(providers.Singleton(model_loader))
    app_container.app.override(providers.Object(app))
    app_container.settings.override(providers.Object(settings))

    # Make sure to wire anything initialized in the packages and modules asked
    app_container.wiring_config.packages.extend(wiring_config.get("packages", []))
    app_container.wiring_config.modules.extend(wiring_config.get("modules", []))
    app_container.wire()

    logger: ILogger = app_container.logger()
    logger.info("Application container bootstrapped")

    logger.info("Registering initialization and shutdown handlers")
    initialize_model_on_startup = model_loader is not None
    app.add_event_handler(
        "startup",
        func=partial(
            on_startup,
            setup_ml_framework=setup_ml_framework or (lambda: None),
            load_model=initialize_model_on_startup,
        ),
    )

    app.add_event_handler("shutdown", on_shutdown)

    config = uvicorn.Config(
        app,
        host=settings.HOST,
        port=settings.SPAICE_APPLICATION_PORT,
        timeout_keep_alive=settings.UVICORN_KEEP_ALIVE_TIMEOUT,
        limit_concurrency=settings.UVICORN_CONCURRENCY,
    )
    server = uvicorn.Server(config)

    # Align logging of uvicorn and tensorflow
    for name in ["uvicorn", "uvicorn.access", "uvicorn.error", "tensorflow"]:
        logging.getLogger(name).handlers = [logger.handlers[0]]
        logging.getLogger(name).propagate = False

    return server
