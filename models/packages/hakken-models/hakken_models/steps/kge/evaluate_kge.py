from typing import Annotated, Any

from torch.utils.data import DataLoader
from zenml import step
from zenml.client import Client

from hakken_models.core.configs.evaluator import EvaluatorConfig
from hakken_models.evaluators.kge import KGEEvaluator
from hakken_models.models.kge import KGE


# TODO: Implement actual evaluation logic
@step(enable_cache=False, experiment_tracker=Client().active_stack.experiment_tracker.name)
def evaluate_kge_step(
    data_loader: DataLoader,
    model: KGE,
    evaluator_config: EvaluatorConfig,
    dataset_metadata: dict,
    split_name: str,
) -> Annotated[list[dict[str, Any]], "{split_name}_metrics_dict"]:
    evaluator = KGEEvaluator(
        metrics_config=evaluator_config.metrics,
        max_num_batches=evaluator_config.max_num_batches,
        num_relations=dataset_metadata.get("num_relations"),
    )

    evaluator.update_from_dataloader(model=model, data_loader=data_loader)

    output = evaluator.compute()

    for metric_dict in output:
        metric_dict["split_name"] = split_name

    return output
