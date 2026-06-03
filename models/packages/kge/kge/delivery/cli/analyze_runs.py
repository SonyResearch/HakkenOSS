import mlflow
import pandas as pd
from loguru import logger
from mlflow.tracking import MlflowClient


def main():
    # TODO: WIP
    # Point MLflow to your local folder
    mlflow.set_tracking_uri("file:.ml_flow")

    # Get runs for an experiment
    client = MlflowClient()

    experiment = mlflow.get_experiment_by_name("pubtator_complex_hpo")
    logger.info(f"Loading runs of experiment {experiment.name}")
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])

    logger.info(f"Found {len(runs)} runs")

    df = pd.DataFrame([r.data.metrics | r.data.params | {"run_id": r.info.run_id} for r in runs])
    df = df.loc[:, df.nunique() > 1]

    logger.info(df.head())


if __name__ == "__main__":
    main()
