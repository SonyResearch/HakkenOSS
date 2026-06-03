import os
import tempfile

import mlflow
import pandas as pd
from loguru import logger
from zenml import step
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker


@step(
    enable_cache=False,
    experiment_tracker=experiment_tracker.name,
)
def log_artifact_csv_step(
    mlflow_run_id: str,
    data: list[dict],
    artifact_path: str,
    filename: str,
    tags: dict[str, str] | None = None,
) -> None:
    with mlflow.start_run(run_id=mlflow_run_id, nested=True):
        df = pd.DataFrame(data)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, filename)
            df.to_csv(path, index=False)
            mlflow.log_artifact(path, artifact_path=artifact_path)
        if tags is not None:
            for key, value in tags.items():
                mlflow.set_tag(key, value)

    logger.info(f"Logged {len(df)} rows → {artifact_path}/{filename} (run: {mlflow_run_id[:8]}...)")
