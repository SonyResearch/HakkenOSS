from typing import Annotated

from hakken_ml_toolkit.ml_utils.extras import FactBatchUtils
from zenml import ArtifactConfig, log_metadata, step

from hakken_models.core.entities.supervised_dataset import SupervisedDataset
from hakken_models.datasets.deployment import DatasetDeployment


@step(enable_cache=True)
def create_sup_dataset_step(
    dataset: DatasetDeployment, split_name: str
) -> Annotated[SupervisedDataset, ArtifactConfig(name="{split_name}_supervised_dataset")]:
    facts_pt = dataset.get_facts_tensor(split_name=split_name)

    entity_pairs, relations = FactBatchUtils.to_so_batch_and_relations(
        facts_pt[:, :3], num_relations=dataset.num_relations
    )

    sup_dataset = SupervisedDataset(entity_pairs=entity_pairs, relations=relations)

    metadata = sup_dataset.get_metadata()
    metadata["split"] = split_name

    log_metadata(
        metadata=metadata,
        infer_artifact=True,
    )

    return sup_dataset
