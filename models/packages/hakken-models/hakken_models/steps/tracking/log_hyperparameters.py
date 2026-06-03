import mlflow
from loguru import logger
from zenml import step
from zenml.client import Client

from hakken_models.core.utils.data import flatten_dict

experiment_tracker = Client().active_stack.experiment_tracker


@step(
    enable_cache=False,
    experiment_tracker=experiment_tracker.name,
)
def log_hyperparameters_step(
    hparams: dict,
    mlflow_run_id: str | None = None,
) -> None:
    """
    Log hyperparameters from a Pydantic model to a specific MLflow run.

    The MLflow run ID is retrieved from metadata of a specific step.

    Args:
        hparams: Pydantic model containing hyperparameters
        step_name: Name of the step to get metadata from
        mlflow_run_id_key: Key to retrieve the MLflow run ID from metadata
        context: ZenML step context for accessing pipeline run
    """
    active = mlflow.active_run()
    if active is None:
        raise RuntimeError("No active MLflow run in evaluate step (unexpected).")
    logger.info("Active MLflow run_id:", active.info.run_id)

    active_run_id = active.info.run_id
    logger.info(f"Active MLflow run_id: {active_run_id}")

    if mlflow_run_id is None:
        mlflow_run_id = active_run_id
        logger.info(f"No run_id provided. Using active run: {mlflow_run_id}")
    elif active_run_id != mlflow_run_id:
        logger.warning(
            f"run_id mismatch. expected={mlflow_run_id} "
            f"active={active_run_id}. Logging to active run."
        )
    # Get metadata from the specified step

    flatten_params = flatten_dict(hparams, sep="/")

    mlflow.log_params(flatten_params)
