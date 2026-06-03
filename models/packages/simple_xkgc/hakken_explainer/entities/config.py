from pathlib import Path
from typing import Literal

from datasets.common.constants import DataSplits
from hakken_ml_toolkit.ml_base_structures import Fact
from kge.common.entities.kge_loader_config import KGELoadExperimentConfig
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings

from hakken_explainer.constants import ScoreType


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


class OutputConfig(BaseSettings):
    path: Path
    delimiter: str

    @field_validator("path")
    @classmethod
    def validate_paths(cls, v: Path | str) -> Path:
        return Path(v)


class RunConfig(BaseModel):
    device: Literal["cpu", "cuda"] = "cuda"


class ScoreTypeConfig(BaseModel):
    type: ScoreType
    batch_size: int

    @field_validator("type", mode="before")
    @classmethod
    def convert_type_to_enum(cls, v):
        if isinstance(v, str):
            return ScoreType(v)
        return v


class Config(BaseSettings):
    triple_to_probe: Fact
    explainer: HakkenExplainerConfig
    output: OutputConfig
    log_level: str
    run: RunConfig
