from os import getenv
from typing import TYPE_CHECKING, ClassVar

from dependency_injector import containers, providers
from typing_extensions import TypedDict

# Actions
from spaice_inference_api.core.action.health_check import HealthCheckAction
from spaice_inference_api.core.action.load_model import LoadModelAction

# Contracts
from spaice_inference_api.core.contract.model import IModel, IModelLoader

# Dispatcher
from spaice_inference_api.core.dispatcher import Dispatcher

# Externals
from spaice_inference_api.impl.logging.logging_impl import Logging
from spaice_inference_api.impl.metrics.metrics import Metrics
from spaice_inference_api.impl.metrics.prometheus_metrics import PrometheusMetrics

if TYPE_CHECKING:
    from fastapi import FastAPI

    from spaice_inference_api.config import Settings
    from spaice_inference_api.core.contract.logger import ILogger
    from spaice_inference_api.core.contract.metrics.inference_metrics import (
        IMetrics as IInferenceMetrics,
    )
    from spaice_inference_api.core.contract.metrics.metrics import IMetrics, MetricLabels


class MetricsConfiguration(TypedDict, total=False):
    common_key_prefix: str
    common_labels: "MetricLabels"


class ContainerConfiguration(TypedDict, total=False):
    model_name: str
    logger_name: str
    metrics: MetricsConfiguration


class Container(containers.DeclarativeContainer):
    wiring_config = containers.WiringConfiguration(packages=["spaice_inference_api"])
    # The following is so that we can pass the container itself
    # as a dependency
    __self__: providers.Self["Container"] = providers.Self()

    config = providers.Configuration(strict=True)

    default_config: ClassVar[ContainerConfiguration] = {
        "model_name": "default_model_name",
        "logger_name": "spaice_inference_api",
        "metrics": {
            "common_key_prefix": getenv(
                "SPAICE_PROMETHEUS_METRICS_PREFIX", "spaice_inference_api_"
            ),
            "common_labels": {},
        },
    }

    config.from_dict(dict(default_config))

    settings: providers.Provider["Settings"] = providers.Object(None)

    # TODO: Add somewhere here the default logger name
    logging: providers.Provider[Logging] = providers.Singleton(Logging)
    logger: providers.Provider["ILogger"] = providers.Object(logging().get_logger("spaice"))
    infra_metrics: providers.Provider["IMetrics"] = providers.Singleton(
        PrometheusMetrics,
        common_key_prefix=config.metrics.common_key_prefix,
        common_labels=config.metrics.common_labels,
    )
    metrics: providers.Provider["IInferenceMetrics"] = providers.Singleton(
        Metrics, metrics=infra_metrics.provided
    )

    ##############
    # DISPATCHER #
    ##############

    dispatcher: providers.Provider[Dispatcher] = providers.Factory(
        Dispatcher,
        actions=providers.Dict(
            {
                "HealthCheckAction": providers.Factory(HealthCheckAction),
                "LoadModelAction": providers.Factory(LoadModelAction),
            }
        ),
    )

    model: providers.Provider[IModel] = providers.Object(None)
    model_loader: providers.Provider[IModelLoader] = providers.AbstractSingleton(IModelLoader)

    app: providers.Provider["FastAPI"] = providers.Object(None)
