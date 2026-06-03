from time import time
from typing import TYPE_CHECKING

from anyio import CapacityLimiter
from anyio.lowlevel import RunVar
from dependency_injector import providers
from dependency_injector.wiring import Provide, inject

from spaice_inference_api.container import Container
from spaice_inference_api.core.contract.logger import ILogger, LoggerToken
from spaice_inference_api.core.contract.model import ModelLoadingOptions

if TYPE_CHECKING:
    from collections.abc import Callable


@inject
def setup_capacity_limiter(container: Container = Provide[Container.__self__]):
    settings = container.settings()
    container.logging().get_logger(__name__).info(
        f'Setting uvicorn limiter at "{settings.UVICORN_CAPACITY_LIMITER}"'
    )
    # The following will limit the number of active threads being processed
    # at the same time. New requests will be queued and be processed once
    # threads from this pool are available
    # https://github.com/tiangolo/fastapi/issues/4221
    RunVar("_default_thread_limiter").set(
        value=CapacityLimiter(total_tokens=settings.UVICORN_CAPACITY_LIMITER)  # type: ignore
    )


@inject
def initialize_model(container: Container = Provide[Container.__self__]):
    settings = container.settings()
    options = ModelLoadingOptions(path=settings.SPAICE_MODEL_PATH)
    model = container.dispatcher().actions["LoadModelAction"].go(options=options)
    container.logger().info("Registering loaded model in container")
    container.model.override(providers.Object(model))


@inject
def on_shutdown(container: Container = Provide[Container.__self__]):
    container.shutdown_resources()


@inject
def on_startup(
    setup_ml_framework: "Callable",
    load_model: bool = False,
    logger: ILogger = Provide[LoggerToken],
):
    startup_start_time = time()
    setup_capacity_limiter()
    if setup_ml_framework is not None:
        setup_ml_framework()
    if load_model:
        initialize_model()
    logger.info(f"Startup completed in {time() - startup_start_time:.2f} seconds")
