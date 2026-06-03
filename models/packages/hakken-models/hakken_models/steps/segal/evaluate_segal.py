"""Step to evaluate a trained SeGAL model."""

from typing import Annotated, Any

from zenml import step
from zenml.client import Client

from hakken_models.core.configs.evaluator import EvaluatorConfig
from hakken_models.data_loaders import TemporalKGLinkNeighborLoader
from hakken_models.datasets.deployment import DatasetDeployment
from hakken_models.evaluators.segal import SeGALEvaluator
from hakken_models.models.segal import SeGAL, SeGALInferenceWrapper

experiment_tracker = Client().active_stack.experiment_tracker


@step(enable_cache=False, experiment_tracker=experiment_tracker.name)
def evaluate_segal_step(
    data_loader: TemporalKGLinkNeighborLoader,
    model: SeGAL,
    dataset: DatasetDeployment,
    evaluator_config: EvaluatorConfig,
    dataset_metadata: dict,
    split_name: str,
) -> Annotated[list[dict[str, Any]], "{split_name}_metrics_dict"]:
    """Evaluate a trained SeGAL model on a split."""
    wrapper = SeGALInferenceWrapper(
        segal=model,
        node_embeddings=dataset.get_node_embedding_matrix(),
        relation_embeddings=dataset.get_relation_embedding_matrix(),
    )
    wrapper.eval()
    device = next(model.parameters()).device
    wrapper = wrapper.to(device)

    evaluator = SeGALEvaluator(
        metrics_config=evaluator_config.metrics,
        max_num_batches=evaluator_config.max_num_batches,
        num_relations=dataset_metadata.get("num_relations"),
    )

    evaluator.update_from_dataloader(model=wrapper, data_loader=data_loader)

    output = evaluator.compute()

    for metric_dict in output:
        metric_dict["split_name"] = split_name

    return output
