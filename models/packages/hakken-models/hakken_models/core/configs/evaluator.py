from typing import Any

from pydantic import BaseModel, Field


class MetricConfig(BaseModel):
    name: str
    target_class: str
    kwargs: dict[str, Any] = Field(default_factory=dict)
    parameter_bindings: dict[str, str] = Field(default_factory=dict)
    prediction_mode: str | None = Field(
        default=None, description="Link prediction direction: 'subject', 'object', or 'relation'."
    )

    divide_by_relation: bool = Field(
        default=False,
        description="Whether to compute the metric separately for each relation type.",
    )

    def resolve_parameters(self, context: dict[str, Any]) -> dict[str, Any]:
        parameters = self.kwargs.copy()

        for constructor_arg, context_key in self.parameter_bindings.items():
            if context_key not in context:
                raise KeyError(
                    f"Metric '{self.name}' requires context value "
                    f"'{context_key}', but it is not available."
                )

            value = context[context_key]
            if value is not None:
                parameters[constructor_arg] = value

        return parameters


class EvaluatorConfig(BaseModel):
    metrics: list[MetricConfig] = Field(
        default_factory=lambda: [
            MetricConfig(name="top_k_10", target_class="hakken_utils.HitsAtK", kwargs={"k": 10})
        ]
    )

    max_num_batches: int = Field(
        default=1_000_000,
        ge=1,
        description=("Maximum number of batches to evaluate. "),
    )
