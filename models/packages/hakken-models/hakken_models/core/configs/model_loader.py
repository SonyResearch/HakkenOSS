from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings


class ModelLoaderConfig(BaseSettings):
    # -------------------
    # Common info
    # -------------------
    device: Literal["cpu", "cuda"] = Field(
        default="cuda", description="Device on which the model will be loaded and run."
    )
    ckpt_is_lightning: bool = Field(
        default=True, description="Whether the checkpoint was saved using PyTorch Lightning."
    )
    relative_ckpt_path: str = Field(
        default="last/last.ckpt",
        description="Path to the checkpoint relative to the run directory or MLflow artifact root.",
    )
    data_root_uri_template: str = Field(
        default="s3://sai-spaice-ds/data/processed/data_processing/zenml/{name}/{version}",
        description="Template for dataset root URI. Supports variables {name} and {version}.",
    )

    # -------------------
    # MODE 1 — MLflow
    # -------------------

    mlflow_run_id: str | None = Field(
        default=None, description="MLflow run ID to load the model from."
    )
    artifact_path: str = Field(
        default="checkpoints", description="Artifact subdirectory inside the MLflow run."
    )
    tracking_uri: str = Field(
        default="s3://hakken-mlflow/mlruns", description="MLflow tracking URI."
    )

    # -------------------
    # MODE 2 — Directory
    # -------------------
    run_dir: str | None = Field(
        default=None, description="Local directory containing the model checkpoint and parameters."
    )
    relative_params_path: str = Field(
        default="params.json", description="Path to the parameters file relative to run_dir."
    )

    # -------------------
    # Param overrides
    # -------------------
    param_overrides: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Flat key/value overrides applied to run params before unflattening. "
            "Keys use the same separator as MLflow params (e.g. 'segal/embedder/base_url')."
        ),
    )

    @field_validator("mlflow_run_id", "run_dir", mode="before")
    @classmethod
    def empty_str_to_none(cls, v: str | None) -> str | None:
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

    @model_validator(mode="after")
    def check_loading_mode(self) -> "ModelLoaderConfig":
        if self.mlflow_run_id and self.run_dir:
            raise ValueError("mlflow_run_id and run_dir are mutually exclusive; provide only one.")

        if not self.mlflow_run_id and not self.run_dir:
            raise ValueError("Either mlflow_run_id or run_dir must be provided.")

        return self

    @property
    def use_mlflow(self) -> bool:
        """Determine whether to load from MLflow or local directory."""
        return self.mlflow_run_id is not None

    @property
    def use_local_dir(self) -> bool:
        """Determine whether to load from local directory."""
        return self.run_dir is not None
