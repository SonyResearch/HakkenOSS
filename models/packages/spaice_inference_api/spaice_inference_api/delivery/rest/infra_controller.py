from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, Response
from fastapi.exceptions import HTTPException

from spaice_inference_api.core.contract.logger import ILogger, LoggerToken
from spaice_inference_api.core.contract.metrics.inference_metrics import (
    IMetrics,
    MetricsToken,
)
from spaice_inference_api.core.dispatcher import Dispatcher, DispatcherToken

router = APIRouter()
LOGGER_DEPENDENCY = Depends(Provide[LoggerToken])
DISPATCHER_DEPENDENCY = Depends(Provide[DispatcherToken])
METRICS_DEPENDENCY = Depends(Provide[MetricsToken])


@router.get("/health-check")
@inject
def health_check(
    logger: ILogger = LOGGER_DEPENDENCY,
    dispatcher: Dispatcher = DISPATCHER_DEPENDENCY,
):
    try:
        pass
        return dispatcher.actions["HealthCheckAction"].go()
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/metrics")
@inject
def metrics(
    logger: ILogger = LOGGER_DEPENDENCY,
    metrics: IMetrics = METRICS_DEPENDENCY,
) -> Response:
    try:
        return Response(
            metrics.report(),
            headers={"Content-Type": "text/plain; version=0.0.4; charset=utf-8"},
        )
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e)) from e
