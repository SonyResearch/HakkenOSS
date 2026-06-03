from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import wraps
from time import time
from typing import Any, TypeVar, cast

from dependency_injector.wiring import Provide, inject
from fastapi import HTTPException
from typing_extensions import ParamSpec

from spaice_inference_api.core.contract.logger import ILogger, LoggerToken

MetricsToken = "metrics"

P = ParamSpec("P")
R = TypeVar("R")
F = TypeVar("F", bound=Callable[..., Any])


class IMetrics(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def new_request(self, path: str, method: str) -> None:
        pass

    @abstractmethod
    def track_model_prediction_time(self, duration: float) -> None:
        pass

    @abstractmethod
    def track_request_time(self, duration: float, path: str, method: str) -> None:
        pass

    @abstractmethod
    def new_model_prediction(self) -> None:
        pass

    @abstractmethod
    def new_model_error(self, error_type: str) -> None:
        pass

    @abstractmethod
    def new_request_error(self, path: str, method: str, code: int, error_type: str) -> None:
        pass

    @abstractmethod
    def report(self) -> bytes:
        pass


def time_model_prediction(func: F) -> F:
    @wraps(func)
    @inject
    def wrapper(
        *args,
        logger: ILogger = Provide[LoggerToken],
        metrics: IMetrics = Provide[MetricsToken],
        **kwargs,
    ):
        try:
            model_prediction_start = time()
            result = func(*args, **kwargs)
            metrics.track_model_prediction_time(time() - model_prediction_start)
            return result
        except Exception as e:
            logger.exception(e)
            error_type = type(e).__name__
            metrics.new_model_error(error_type=error_type)
            raise HTTPException(status_code=500, detail=str(e)) from e
        finally:
            metrics.new_model_prediction()

    return cast("F", wrapper)
