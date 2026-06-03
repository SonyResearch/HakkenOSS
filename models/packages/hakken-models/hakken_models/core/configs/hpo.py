from pydantic import BaseModel, Field


class TuneReporterConfig(BaseModel):
    metrics: str | list[str] | dict[str, str] | None = None
    filename: str = "checkpoint"
    save_checkpoints: bool = False
    on: str | list[str] = "validation_end"


class MetricConfig(BaseModel):
    name: str
    mode: str = "min"


class HPOConfig(BaseModel):
    """Configuration for hyperparameter tuning."""

    reporter: TuneReporterConfig = Field(default_factory=TuneReporterConfig)

    resources_per_trial: dict[str, int] = Field(default_factory=lambda: {"cpu": 1, "gpu": 1})

    metric: MetricConfig = Field(
        default_factory=lambda: MetricConfig(name="val/mean_rank", mode="min"),
        description="The metric to optimize during HPO",
    )
    num_trials: int = Field(default=10, description="Number of HPO trials to run")
    name: str = Field(default="my_hpo_experiment", description="Name of the HPO experiment")
    relative_storage_path: str = Field(
        default="./ray_results/hakken_hpo", description="Storage path for Ray Tune results"
    )
    verbose: int = Field(default=1, description="Verbosity level for Ray Tune")

    def build_search_space(self) -> dict[str, any]:
        return {k: v.build() for k, v in self.search_space.items()}
