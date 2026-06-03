from typing import Annotated

from zenml import ArtifactConfig, log_metadata, step

from hakken_models.core.entities.kg_data import KGData
from hakken_models.datasets.deployment import DatasetDeployment


@step(enable_cache=False)
def load_kg_data_step(
    dataset: DatasetDeployment,
    split_names: list[str],
) -> Annotated[KGData, ArtifactConfig(name="{split_name}_kg_data")]:
    kg_data = dataset.get_kg_data(split_names=split_names)
    metadata = kg_data.get_metadata()
    metadata["split_names"] = split_names
    log_metadata(metadata=metadata, infer_artifact=True)
    return kg_data
