from typing import Annotated, Any

from zenml import step
from zenml.client import Client

from hakken_models.core.configs.evaluator import EvaluatorConfig
from hakken_models.data_loaders import KGLinkNeighborLoader
from hakken_models.evaluators.thiger import THiGEREvaluator
from hakken_models.models.thiger import THiGER

experiment_tracker = Client().active_stack.experiment_tracker


@step(enable_cache=False, experiment_tracker=experiment_tracker.name)
def evaluate_thiger_step(
    data_loader: KGLinkNeighborLoader,
    model: THiGER,
    evaluator_config: EvaluatorConfig,
    dataset_metadata: dict,
    split_name: str,
) -> Annotated[list[dict[str, Any]], "{split_name}_metrics_dict"]:
    evaluator = THiGEREvaluator(
        metrics_config=evaluator_config.metrics,
        max_num_batches=evaluator_config.max_num_batches,
        num_relations=dataset_metadata.get("num_relations"),
    )

    evaluator.update_from_dataloader(model=model, data_loader=data_loader)

    output = evaluator.compute()

    for metric_dict in output:
        metric_dict["split_name"] = split_name

    return output
