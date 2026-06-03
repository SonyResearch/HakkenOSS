from pydantic import BaseModel, Field


class EarlyStoppingConfig(BaseModel):
    monitor: str = Field(
        default="val_loss", description="Metric to monitor (must be logged with self.log(...))."
    )
    mode: str = Field(
        default="min", description="One of 'min' (for loss) or 'max' (for accuracy/F1/etc.)."
    )
    patience: int = Field(
        default=5, description="Number of validation checks with no improvement before stopping."
    )
    min_delta: float = Field(
        default=0.0, description="Minimum change to qualify as improvement (absolute)."
    )
    verbose: bool = Field(
        default=True, description="Print message when early stopping is triggered."
    )
    strict: bool = Field(default=True, description="Raise error if monitored metric is not found.")
    check_finite: bool = Field(
        default=True, description="Stop if monitored metric becomes NaN or infinite."
    )
    # Less frequently changed but still useful:
    stopping_threshold: float | None = Field(
        default=None, description="Immediately stop once metric reaches/improves beyond this value."
    )
    divergence_threshold: float | None = Field(
        default=None, description="Stop if metric becomes worse than this value."
    )
