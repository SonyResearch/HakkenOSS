from typing import TYPE_CHECKING, ClassVar

from prometheus_client import Counter, Gauge, Histogram
from prometheus_client import generate_latest as prom_generate_latest

from spaice_inference_api.core.contract.metrics.metrics import (
    ICounter,
    IGauge,
    IHistogram,
    IMetrics,
    Metric,
    MetricLabels,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


class PrometheusMetrics(IMetrics):
    """
    PrometheusMetrics is a class for managing and exporting Prometheus metrics.

    Attributes:
        __counters__ (Dict[str, Counter]): Dictionary to store Prometheus Counter objects.
        __gauges__ (Dict[str, Gauge]): Dictionary to store Prometheus Gauge objects.
        __histograms__ (Dict[str, Histogram]): Dictionary to store Prometheus Histogram objects.
        __common_key_prefix__ (str): Common prefix for metric keys.
        __common_labels__ (MetricLabels): Common labels for all metrics.
    """

    __counters__: ClassVar[dict[str, Counter]] = {}
    __gauges__: ClassVar[dict[str, Gauge]] = {}
    __histograms__: ClassVar[dict[str, Histogram]] = {}

    __common_key_prefix__: str = ""
    __common_labels__: MetricLabels

    def __init__(self, common_key_prefix: str = "", common_labels: MetricLabels | None = None):
        """
        Initialize the PrometheusMetrics with a common key prefix and common labels.

        Args:
            common_key_prefix (str, optional): Common prefix for metric keys. Defaults to "".
            common_labels (MetricLabels, optional): Common labels for all metrics. Defaults to {}.
        """
        if common_labels is None:
            common_labels = {}
        self.__common_labels__ = common_labels
        self.__common_key_prefix__ = common_key_prefix

    def __get_description__(self, metric: Metric, provided_description: str = "") -> str:
        """
        Get the description for a metric, using a provided description or the metric's description.

        Args:
            metric (Metric): The metric for which to get the description.
            provided_description (str, optional): The provided description. Defaults to "".

        Returns:
            str: The description for the metric.
        """
        """Returns the default description if using a predefined metric key"""
        if (
            provided_description != ""
            or not hasattr(metric, "description")
            or metric.description == ""
        ):
            return provided_description

        description = metric.description
        if isinstance(description, str):
            return description

        return ""

    def __get_key__(self, key: str) -> str:
        """Returns the metric key with the common prefix"""
        return self.__common_key_prefix__ + key

    def __get_labels__(self, provided_labels: MetricLabels | None = None) -> MetricLabels:
        """Returns the labels along with the common labels"""
        if provided_labels is None:
            provided_labels = {}
        return {**self.__common_labels__, **provided_labels}

    def counter(
        self, metric: Metric, provided_labels: MetricLabels | None = None, description=""
    ) -> ICounter:
        """
        Get or create a Prometheus Counter metric.

        Args:
            metric (Metric): The metric definition.
            provided_labels (MetricLabels, optional): Labels to attach to the metric. Defaults to
                {}.
            description (str, optional): Description of the metric. Defaults to "".

        Returns:
            ICounter: The Prometheus Counter metric.
        """
        if provided_labels is None:
            provided_labels = {}
        provided_key = metric.value
        key = self.__get_key__(provided_key)
        labels = self.__get_labels__(provided_labels)

        if key not in self.__counters__:
            self.__counters__[key] = Counter(
                key, self.__get_description__(metric, description), labels.keys()
            )

        if len(labels) > 0:
            return self.__counters__[key].labels(*labels.values())

        return self.__counters__[key]

    def gauge(
        self, metric: Metric, provided_labels: MetricLabels | None = None, description=""
    ) -> IGauge:
        """
        Get or create a Prometheus Gauge metric.

        Args:
            metric (Metric): The metric definition.
            provided_labels (MetricLabels, optional): Labels to attach to the metric. Default: {}.
            description (str, optional): Description of the metric. Defaults to "".

        Returns:
            IGauge: The Prometheus Gauge metric.
        """
        if provided_labels is None:
            provided_labels = {}
        provided_key = metric.value
        key = self.__get_key__(provided_key)
        labels = self.__get_labels__(provided_labels)

        if key not in self.__gauges__:
            self.__gauges__[key] = Gauge(
                key, self.__get_description__(metric, description), labels.keys()
            )

        if len(labels) > 0:
            return self.__gauges__[key].labels(*labels.values())

        return self.__gauges__[key]

    def histogram(
        self,
        metric: Metric,
        provided_labels: MetricLabels | None = None,
        buckets: "Sequence[float | str]" = Histogram.DEFAULT_BUCKETS,
        description="",
    ) -> IHistogram:
        """
        Get or create a Prometheus Histogram metric.

        Args:
            metric (Metric): The metric definition.
            provided_labels (MetricLabels, optional): Labels to attach to the metric. Default: {}.
            buckets (Sequence[Union[float, str]], optional): Buckets for the histogram. Default:
                Histogram.DEFAULT_BUCKETS.
            description (str, optional): Description of the metric. Defaults to "".

        Returns:
            IHistogram: The Prometheus Histogram metric.
        """
        if provided_labels is None:
            provided_labels = {}
        provided_key = metric.value
        key = self.__get_key__(provided_key)
        labels = self.__get_labels__(provided_labels)

        if key not in self.__histograms__:
            self.__histograms__[key] = Histogram(
                key,
                self.__get_description__(metric, description),
                labels.keys(),
                buckets=buckets,
            )

        if len(labels) > 0:
            return self.__histograms__[key].labels(*labels.values())

        return self.__histograms__[key]

    def report(self) -> bytes:
        """
        Generate the latest metrics data in Prometheus format.

        Returns:
            bytes: The latest metrics data in Prometheus format.
        """
        return prom_generate_latest()
