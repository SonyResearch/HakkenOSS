from datetime import datetime

from zenml import pipeline

from hakken_models.core.configs.evaluate_model import EvaluateModelConfig
from hakken_models.core.configs.zenml import ContainerSettings, KubernetesKind, OrchestratorSettings
from hakken_models.steps.dataset import extract_dataset_metadata_step
from hakken_models.steps.kge import evaluate_kge_step, load_dataloader_step, load_kge_artifacts_step
from hakken_models.steps.tracking import log_artifact_csv_step


@pipeline(
    enable_cache=True,
    settings={
        "orchestrator": OrchestratorSettings.kubernetes(KubernetesKind.IN_CLUSTER),
        "docker": ContainerSettings.docker(),
    },
)
def evaluate_kge_pipeline(config: EvaluateModelConfig) -> list[dict[str, float]]:
    """Evaluate a trained KGE model on specified dataset splits."""

    model, dataset = load_kge_artifacts_step(config.model_loader)

    dataset_metadata = extract_dataset_metadata_step(dataset=dataset)

    now = datetime.now().strftime("%Y%m%d_%H%M%S")

    for group_name, split_names in config.evaluation_groups.items():
        data_loader = load_dataloader_step.with_options(substitutions={"split_name": group_name})(
            dataset=dataset,
            split_names=split_names,
            data_loader_config=config.data_loader,
        )

        metrics = evaluate_kge_step.with_options(substitutions={"split_name": group_name})(
            data_loader=data_loader,
            model=model,
            evaluator_config=config.evaluator,
            dataset_metadata=dataset_metadata,
            split_name=group_name,
        )

        log_artifact_csv_step(
            mlflow_run_id=config.model_loader.mlflow_run_id,
            data=metrics,
            artifact_path=f"evaluation/{now}",
            filename=f"{group_name}_metrics.csv",
            tags={"has_evaluation": "true"},
        )

    return metrics
