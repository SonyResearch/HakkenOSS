from zenml import pipeline

from hakken_models.core.configs.evaluate_model import EvaluateModelConfig
from hakken_models.core.configs.zenml import ContainerSettings, KubernetesKind, OrchestratorSettings
from hakken_models.steps.dataset import (
    create_sup_dataset_step,
    extract_dataset_metadata_step,
    load_kg_data_step,
)
from hakken_models.steps.thiger import (
    evaluate_thiger_step,
    load_thiger_artifacts_step,
    load_thiger_dataloader_step,
)


@pipeline(
    enable_cache=True,
    settings={
        "orchestrator": OrchestratorSettings.kubernetes(KubernetesKind.IN_CLUSTER),
        "docker": ContainerSettings.docker(),
    },
)
def evaluate_thiger_pipeline(config: EvaluateModelConfig) -> list[dict[str, float]]:
    """Evaluate a trained THiGER model on specified dataset splits.

    This pipeline:
    1. Loads a trained THiGER model from MLflow or directory
    2. Loads dataset metadata and evaluation data for specified splits
    3. Creates data loaders for each split
    4. Runs evaluation and returns metrics

    Args:
        config: Evaluation configuration specifying model source and
            evaluation settings. Dataset configuration is inferred from the model.

    Returns:
        list of dictionaries with metrics results.
    """

    model, dataset = load_thiger_artifacts_step(config.model_loader)

    dataset_metadata = extract_dataset_metadata_step(dataset=dataset)

    for group_name, split_names in config.evaluation_groups.items():
        sup_dataset = create_sup_dataset_step.with_options(
            substitutions={"split_name": group_name}
        )(
            dataset=dataset,
            split_name=group_name,
        )

        kg_data = load_kg_data_step.with_options(substitutions={"split_name": group_name})(
            dataset=dataset,
            split_names=split_names,
        )

        data_loader = load_thiger_dataloader_step.with_options(
            substitutions={"split_name": group_name}
        )(
            kg_data=kg_data,
            sup_dataset=sup_dataset,
            data_loader_config=config.data_loader,
            shuffle=False,
        )

        metrics = evaluate_thiger_step.with_options(substitutions={"split_name": group_name})(
            data_loader=data_loader,
            model=model,
            evaluator_config=config.evaluator,
            dataset_metadata=dataset_metadata,
            split_name=group_name,
        )

    return metrics
