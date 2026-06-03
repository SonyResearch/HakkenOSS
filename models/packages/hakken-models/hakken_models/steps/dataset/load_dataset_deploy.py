from zenml import step

from hakken_models.core.configs.train_common import DatasetConfig
from hakken_models.datasets.deployment import DatasetDeployment


@step
def load_dataset_deployment_step(config: DatasetConfig) -> DatasetDeployment:
    """Create a DatasetDeployment from the provided dataset configuration."""

    return DatasetDeployment(
        target_root=config.data_root_uri,
        load_embeddings=config.load_embeddings,
    )
