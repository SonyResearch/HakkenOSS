from pydantic import BaseModel


class ExperimentTrackerConfig(BaseModel):
    experiment_name: str = "test-experiment"
    run_name: str | None = None
