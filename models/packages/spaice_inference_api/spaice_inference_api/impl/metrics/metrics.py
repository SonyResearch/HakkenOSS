from spaice_inference_api.core.contract.metrics.inference_metrics import IMetrics
from spaice_inference_api.core.contract.metrics.metrics import (
    IMetrics as ExternalMetrics,
)
from spaice_inference_api.core.contract.metrics.metrics import (
    IMetricsType,
    MetricLabels,
)


class ModelMetricsType(IMetricsType):
    # Model metrics
    ModelErrors = "model_prediction_error_total", "Model predict error"
    ModelNumberOfRequests = (
        "model_prediction_total",
        "How many requests have been processed?",
    )
    ModelPredictionDuration = (
        "model_prediction_duration_seconds",
        "How much time did the actual prediction take",
    )

    # Server metrics
    HttpRequestErrors = "http_request_error_total", "Endpoint errors"
    HttpRequestNumberOfRequests = (
        "http_request_total",
        "How many requests have been received?",
    )
    HttpRequestDuration = (
        "http_request_duration_seconds",
        "How much time did the actual request take",
    )


BUCKETS = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30, 60]


class Metrics(IMetrics):
    __metrics: ExternalMetrics

    def __init__(self, metrics: ExternalMetrics):
        self.__metrics = metrics

    def __increase_counter(
        self, metric: ModelMetricsType, labels: MetricLabels | None = None
    ) -> None:
        if labels is None:
            labels = {}
        self.__metrics.counter(metric, labels).inc()

    def __track_time(
        self, duration: float, metric: ModelMetricsType, labels: MetricLabels | None = None
    ) -> None:
        if labels is None:
            labels = {}
        if metric is None:
            raise Exception("There is no such metric to track")
        self.__metrics.histogram(metric, labels, buckets=BUCKETS).observe(duration)

    def new_request(self, path: str, method: str) -> None:
        self.__increase_counter(
            ModelMetricsType.HttpRequestNumberOfRequests,
            {"endpoint": path, "method": method},
        )

    def new_model_prediction(self) -> None:
        self.__increase_counter(ModelMetricsType.ModelNumberOfRequests, {})

    def track_model_prediction_time(self, duration: float) -> None:
        self.__track_time(duration, ModelMetricsType.ModelPredictionDuration)

    def track_request_time(self, duration: float, path: str, method: str) -> None:
        self.__track_time(
            duration,
            ModelMetricsType.HttpRequestDuration,
            {"endpoint": path, "method": method},
        )

    def new_model_error(self, error_type: str) -> None:
        self.__increase_counter(ModelMetricsType.ModelErrors, {"type": error_type})

    def new_request_error(self, path: str, method: str, code: int, error_type: str) -> None:
        self.__increase_counter(
            ModelMetricsType.HttpRequestErrors,
            {"endpoint": path, "method": method, "code": code, "type": error_type},
        )

    def report(self) -> bytes:
        return self.__metrics.report()
