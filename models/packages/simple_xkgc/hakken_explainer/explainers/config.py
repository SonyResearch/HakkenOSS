from pathlib import Path

from datasets.common.constants import DataSplits
from kge.common.entities.kge_loader_config import KGELoadExperimentConfig
from pydantic import Field
from pydantic_settings import BaseSettings


class HakkenExplainerConfig(BaseSettings):
    data_path: Path
    gnn_experiment_config: KGELoadExperimentConfig
    search_space_split_names: list[DataSplits] = Field(
        default=[DataSplits.TRAIN],
        description="Fact splits that are part of the search space",
    )
    graph_cache_folder: Path | None = Field(
        default=None,
        description="If set, the graph will be cached in this folder",
    )
