"""Utilities for managing collections of metrics.

This module provides lightweight abstractions for defining metrics through
configuration and updating them in a coordinated way.

The design separates:

• Metric configuration (`MetricBundle`)
• Runtime metric aggregation (`MetricHub`)

A `MetricBundle` defines how to instantiate a metric and how to map runtime
inputs to the metric's expected arguments.

A `MetricHub` coordinates updating, computing, and resetting multiple metrics.

The implementation is framework-agnostic and works with any object implementing
the :class:`MetricLike` protocol.
"""

from collections import Counter
from typing import Any

import torch
from pydantic import BaseModel, Field

from hakken_models.evaluators.metric_bundle import MetricBundle


class MetricHubConfig(BaseModel):
    """Hydra- and settings-friendly spec for building a :class:`MetricHub` (e.g. validation metrics)."""

    enabled: bool = Field(
        default=True,
        description="When False, skip attaching a metric hub (e.g. LitKGE ``val_metric_hub=None``).",
    )
    bundles: list[dict[str, Any]] | None = Field(
        default=None,
        description=(
            "List of kwargs for :class:`~hakken_models.evaluators.metric_bundle.MetricBundle`. "
            "If None and enabled, callers such as LitKGE use the default sampled mean-rank hub. "
            "If a non-empty list, build the hub from these specs. "
            "An empty list yields an empty hub. "
            "For KGE, ``metric_kwargs.num_labels: -1`` is replaced by ``num_relations`` when "
            "building the hub from training metadata."
        ),
    )


class MetricHub:
    """Container managing multiple metrics.

    Responsibilities
    ----------------
    • Update metrics with batch data
    • Compute aggregated metric values
    • Reset metric state
    • Move metrics across devices

    This class is framework-agnostic and can be used in training,
    validation, testing, or offline evaluation pipelines.
    """

    def __init__(self, metric_bundles: list[MetricBundle]) -> None:
        """Raises
        ------
        ValueError
            If two or more bundles share the same ``name`` (``compute()`` would
            otherwise drop values silently).
        """
        name_counts = Counter(b.name for b in metric_bundles)
        duplicates = sorted(name for name, count in name_counts.items() if count > 1)
        if duplicates:
            raise ValueError(f"MetricHub requires unique bundle names; duplicates: {duplicates}")
        self.metric_bundles = metric_bundles

    def update(self, **kwargs: Any) -> None:
        """Update all metrics with the provided batch data."""
        for bundle in self.metric_bundles:
            bundle.update(**kwargs)

    def compute(self) -> dict[str, Any]:
        """Compute all metrics.

        Returns
        -------
        dict[str, Any]
            Mapping of metric name to computed value.
        """
        return {bundle.name: bundle.compute() for bundle in self.metric_bundles}

    def reset(self) -> None:
        """Reset all metric states."""
        for bundle in self.metric_bundles:
            bundle.reset()

    def compute_and_reset(self) -> dict[str, Any]:
        """Compute all metrics and reset their states."""
        results = self.compute()
        self.reset()
        return results

    def to(self, device: torch.device | str) -> "MetricHub":
        """Move all metrics to a device."""
        for bundle in self.metric_bundles:
            bundle.to(device)
        return self
