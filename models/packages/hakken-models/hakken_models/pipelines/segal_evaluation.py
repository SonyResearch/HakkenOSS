"""SeGAL evaluation pipeline."""

from datetime import datetime

from zenml import pipeline

from hakken_models.core.configs.evaluate_model import EvaluateModelConfig
from hakken_models.core.configs.zenml import ContainerSettings, KubernetesKind, OrchestratorSettings
from hakken_models.steps.dataset import extract_dataset_metadata_step, load_kg_data_step
from hakken_models.steps.segal import (
    evaluate_segal_step,
    load_segal_artifacts_step,
    load_segal_dataloader_step,
)
from hakken_models.steps.tracking import log_artifact_csv_step


@pipeline(
    enable_cache=True,
    settings={
        "orchestrator": OrchestratorSettings.kubernetes(KubernetesKind.IN_CLUSTER),
        "docker": ContainerSettings.docker(),
    },
)
def evaluate_segal_pipeline(config: EvaluateModelConfig) -> list[dict[str, float]]:
    """Evaluate a trained SeGAL model on specified dataset splits.

    This pipeline:
    1. Loads a trained SeGAL model from MLflow or directory
    2. Loads dataset metadata and evaluation data for specified splits
    3. Creates TemporalKGLinkNeighborLoader for each split
    4. Runs evaluation and returns metrics
    """
    model, dataset = load_segal_artifacts_step(config.model_loader)

    dataset_metadata = extract_dataset_metadata_step(dataset=dataset)

    now = datetime.now().strftime("%Y%m%d_%H%M%S")

    for group_name, split_names in config.evaluation_groups.items():
        kg_data = load_kg_data_step.with_options(substitutions={"split_name": group_name})(
            dataset=dataset,
            split_names=split_names,
        )

        data_loader = load_segal_dataloader_step.with_options(
            substitutions={"split_name": group_name}
        )(
            kg_data=kg_data,
            dataset=dataset,
            split_name=group_name,
            data_loader_config=config.data_loader,
            num_negatives=32,
            shuffle=False,
        )

        metrics = evaluate_segal_step.with_options(substitutions={"split_name": group_name})(
            data_loader=data_loader,
            model=model,
            dataset=dataset,
            evaluator_config=config.evaluator,
            dataset_metadata=dataset_metadata,
            split_name=group_name,
        )

        if config.model_loader.mlflow_run_id is not None:
            log_artifact_csv_step(
                mlflow_run_id=config.model_loader.mlflow_run_id,
                data=metrics,
                artifact_path=f"evaluation/{now}",
                filename=f"{group_name}_metrics.csv",
                tags={"has_evaluation": "true"},
            )

    return metrics
