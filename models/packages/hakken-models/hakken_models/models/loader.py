import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, TypeAlias, TypeVar

import fsspec
import mlflow
import torch
from loguru import logger

from hakken_models.core.configs.model_loader import ModelLoaderConfig
from hakken_models.core.utils.data import unflatten_dict

DEVICE_TYPE: TypeAlias = torch.device | str | None

ModelArtifacts = TypeVar("ModelArtifacts")
ModelType = TypeVar("ModelType")


class ModelLoader(ABC, Generic[ModelArtifacts, ModelType]):
    """
    Abstract base class for loading trained models and their associated artifacts.

    A model can be loaded either from:
    - an MLflow run (via run_id), or
    - a local or remote directory (via run_dir)

    Subclasses must implement `from_params`, which defines how a concrete model
    and its artifacts are instantiated from configuration parameters and a
    checkpoint path.
    """

    def __init__(self, config: ModelLoaderConfig):
        self.config = config

    @property
    def mlflow_run_id(self) -> str:
        """
        Return the MLflow run ID used for loading the model.

        Returns:
            The MLflow run ID.

        Raises:
            ValueError: If a local run directory is configured instead of MLflow.
        """
        if self.config.mlflow_run_id is None:
            raise ValueError("mlflow_run_id is not specified, no MLflow run_id available.")
        return self.config.mlflow_run_id

    @property
    def dir_uri(self) -> str:
        """
        Return the directory URI used for loading the model.

        Returns:
            The directory URI (local or remote).

        Raises:
            ValueError: If no run directory is configured.
        """
        if self.config.run_dir is None:
            raise ValueError("run_dir is not specified, no directory URI available.")
        return self.config.run_dir

    @abstractmethod
    def from_params(self, params: dict[str, Any], ckpt_path: str) -> ModelArtifacts:
        """
        Construct model artifacts from parameters and a checkpoint.

        Args:
            params: Hierarchical configuration parameters used to build the model.
            ckpt_path: Path or URI to the model checkpoint.

        Returns:
            Loaded model artifacts.

        Notes:
            This method must be implemented by subclasses and defines how a
            specific model type is instantiated.
        """
        pass

    def _apply_flat_overrides(self, flat_params: dict[str, str]) -> dict[str, str]:
        """Apply param_overrides from config onto flat_params (mutates in place)."""
        if self.config.param_overrides:
            logger.info(f"Applying param overrides: {self.config.param_overrides}")
            flat_params.update(self.config.param_overrides)
        return flat_params

    def load(self) -> ModelArtifacts:
        """
        Load a trained model and its artifacts.

        This is a convenience method that dispatches to either `from_mlflow`
        or `from_dir` depending on the configuration.

        Returns:
            Loaded model artifacts.

        Raises:
            ValueError: If neither MLflow nor a directory-based source is configured.
        """
        if self.config.use_local_dir:
            return self.from_dir()

        if self.config.use_mlflow:
            return self.from_mlflow()

        raise ValueError("Must provide either run_id or dir_uri to load the model.")

    def from_mlflow(self) -> ModelArtifacts:
        """
        Load model artifacts from an MLflow run.

        This method:
        1. Retrieves run parameters from MLflow.
        2. Converts flat MLflow parameters into a hierarchical dictionary.
        3. Downloads the configured artifact directory.
        4. Loads the model checkpoint.
        5. Instantiates the model via `from_params`.

        Returns:
            Loaded model artifacts.

        Raises:
            RuntimeError: If the MLflow run contains no parameters.
        """
        if self.config.tracking_uri is not None:
            mlflow.set_tracking_uri(self.config.tracking_uri)

        run = mlflow.get_run(run_id=self.mlflow_run_id)
        logger.info(f"Loading model from MLflow run_id {self.mlflow_run_id}")

        flat_params = run.data.params
        if len(flat_params) == 0:
            raise RuntimeError("No params in run (unexpected).")

        # remove MLflow bookkeeping params if present
        flat_params.pop("last_checkpoint_path", None)
        flat_params.pop("checkpoint_dir", None)

        self._apply_flat_overrides(flat_params)
        params = unflatten_dict(flat_params, sep="/")

        local_dir = mlflow.artifacts.download_artifacts(
            run_id=self.mlflow_run_id,
            artifact_path=self.config.artifact_path,
        )
        logger.info(f"Artifacts downloaded to: {local_dir}")

        ckpt_path = Path(local_dir) / Path(self.config.relative_ckpt_path)
        logger.info(f"Loading checkpoint from: {ckpt_path}")

        return self.from_params(params=params, ckpt_path=str(ckpt_path))

    def from_dir(self) -> ModelArtifacts:
        """
        Load model artifacts from a local or remote directory.

        This method:
        1. Reads a parameters JSON file.
        2. Converts flat parameters into a hierarchical dictionary.
        3. Resolves the checkpoint path.
        4. Instantiates the model via `from_params`.

        Returns:
            Loaded model artifacts.
        """
        logger.info(f"Loading model from directory: {self.dir_uri}")

        # -------------------------
        # 1) load params.json
        # -------------------------
        params_path = f"{self.dir_uri.rstrip('/')}/{self.config.relative_params_path}"
        logger.info(f"Reading params from: {params_path}")

        with fsspec.open(params_path, "r") as f:
            flat_params: dict = json.load(f)

        # remove MLflow bookkeeping params if present
        flat_params.pop("last_checkpoint_path", None)
        flat_params.pop("checkpoint_dir", None)

        self._apply_flat_overrides(flat_params)
        params = unflatten_dict(flat_params, sep="/")

        # -------------------------
        # 2) load checkpoint
        # -------------------------
        ckpt_uri = f"{self.dir_uri.rstrip('/')}/{self.config.relative_ckpt_path.lstrip('/')}"
        logger.info(f"Loading checkpoint from: {ckpt_uri}")

        return self.from_params(params=params, ckpt_path=ckpt_uri)
