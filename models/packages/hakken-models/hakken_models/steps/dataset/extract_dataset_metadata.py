from typing import Any

from zenml import step

from hakken_models.datasets.deployment import DatasetDeployment


@step
def extract_dataset_metadata_step(
    dataset: DatasetDeployment,
) -> dict[str, Any]:
    return {
        "num_entities": dataset.num_entities,
        "num_relations": dataset.num_relations,
        "num_domains": dataset.num_domains,
        "num_timestamps": dataset.num_timestamps,
    }
