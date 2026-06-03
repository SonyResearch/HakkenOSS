from typing import Any

import torch
from loguru import logger
from pydantic import BaseModel, Field, PrivateAttr

from hakken_models.core.utils.runtime import instantiate_from_string
from hakken_models.evaluators.batch_selector import BatchSelectorLike, SelectionLike
from hakken_models.evaluators.metric import MetricLike


class MetricBundle(BaseModel):
    name: str = Field(description="Name of the metric")
    metric_class: str = Field(description="Class path of the metric")
    metric_kwargs: dict[str, Any] = Field(default_factory=dict)
    input_bindings: dict[str, str] = Field(default_factory=dict)

    selector_class: str | None = Field(
        default=None,
        description="Optional class path of a batch selector",
    )
    selector_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword arguments for the selector",
    )
    skip_if_missing_inputs: bool = Field(
        default=False,
        description=(
            "If True, skip this bundle's update when required input_bindings keys are "
            "absent or bound values are None (e.g. no relation labels in the batch)."
        ),
    )

    _metric: MetricLike | None = PrivateAttr(default=None)
    _selector: BatchSelectorLike | None = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        self._metric = instantiate_from_string(
            self.metric_class,
            expected_type=MetricLike,
            **self.metric_kwargs,
        )

        if self.selector_class is not None:
            self._selector = instantiate_from_string(
                self.selector_class,
                expected_type=BatchSelectorLike,
                **self.selector_kwargs,
            )

    @property
    def metric(self) -> MetricLike:
        if self._metric is None:
            raise RuntimeError(f"Metric '{self.name}' has not been instantiated.")
        return self._metric

    @property
    def selector(self) -> BatchSelectorLike | None:
        return self._selector

    def resolve_inputs(self, **kwargs: Any) -> dict[str, Any]:
        expected = set(self.input_bindings)
        provided = set(kwargs)

        missing = expected - provided
        if missing:
            raise ValueError(
                f"Metric '{self.name}' missing required inputs: {missing}. "
                f"Expected inputs: {expected}"
            )

        return {
            metric_input: kwargs[source_input]
            for source_input, metric_input in self.input_bindings.items()
        }

    def update(self, **kwargs: Any) -> None:
        try:
            inputs = self.resolve_inputs(**kwargs)
        except ValueError:
            if self.skip_if_missing_inputs:
                logger.warning(f"Skipping metric '{self.name}' update due to missing inputs.")
                return
            raise
        if self.skip_if_missing_inputs and any(v is None for v in inputs.values()):
            logger.warning(f"Skipping metric '{self.name}' update due to missing inputs.")
            return
        selected_inputs = self.select_inputs(inputs, **kwargs)
        if selected_inputs is None:
            return
        self.metric.update(**selected_inputs)

    def compute(self) -> Any:
        return self.metric.compute()

    def reset(self) -> None:
        self.metric.reset()

    def to(self, device: torch.device | str) -> "MetricBundle":
        self.metric.to(device)
        return self

    def select_inputs(
        self,
        inputs: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        if self.selector is None:
            return inputs

        selection = self.selector(**kwargs)
        filtered = {key: self._apply_selection(value, selection) for key, value in inputs.items()}

        if self._is_empty_selection(filtered):
            return None

        return filtered

    @staticmethod
    def _apply_selection(value: Any, selection: SelectionLike) -> Any:
        if isinstance(value, torch.Tensor):
            return value[selection]
        if isinstance(value, list):
            if isinstance(selection, slice):
                return value[selection]
            if isinstance(selection, torch.Tensor):
                if selection.dtype == torch.bool:
                    selection = selection.tolist()
                    return [v for v, keep in zip(value, selection, strict=False) if keep]
                selection = selection.tolist()
            if selection and isinstance(selection[0], bool):
                return [v for v, keep in zip(value, selection, strict=False) if keep]
            return [value[i] for i in selection]
        if isinstance(value, tuple):
            if isinstance(selection, slice):
                return value[selection]
            if isinstance(selection, torch.Tensor):
                if selection.dtype == torch.bool:
                    selection = selection.tolist()
                    return tuple(v for v, keep in zip(value, selection, strict=False) if keep)
                selection = selection.tolist()
            if selection and isinstance(selection[0], bool):
                return tuple(v for v, keep in zip(value, selection, strict=False) if keep)
            return tuple(value[i] for i in selection)

        return value

    @staticmethod
    def _is_empty_selection(inputs: dict[str, Any]) -> bool:
        for value in inputs.values():
            if isinstance(value, torch.Tensor):
                return value.shape[0] == 0
            if isinstance(value, (list, tuple)):
                return len(value) == 0
        return False
