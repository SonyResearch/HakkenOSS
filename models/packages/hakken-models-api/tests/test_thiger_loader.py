import types
from unittest.mock import MagicMock

import pytest

from hakken_models_api.loaders.thiger import THiGERRunLoader
from hakken_models.models.thiger import THiGERLoader
from hakken_models_api.config import HakkenModelsAPIConfig
from spaice_inference_api import  ModelLoadingOptions


def test_thiger_run_loader_load(monkeypatch: pytest.MonkeyPatch):
    # --- Arrange: fake return value from THiGERLoader ---
    fake_artifacts = object()

    mocked_from_mlflow = MagicMock(return_value=fake_artifacts)
    monkeypatch.setattr(THiGERLoader, "from_mlflow", mocked_from_mlflow)

    # Arrange
    config = HakkenModelsAPIConfig(
        mlflow_run_id="run123",
        artifact_path="artifacts",
        tracking_uri="http://mlflow",
        relative_ckpt_path="ckpt.ckpt",
        ckpt_is_lightning=True,
        device="cpu",
    )

    logger = MagicMock()

    loader = THiGERRunLoader()

    # Act
    result = loader.load(options=ModelLoadingOptions(path="path/to/model"), logger=logger, config=config)

    # Assert
    assert result is fake_artifacts
    mocked_from_mlflow.assert_called_once_with(
        run_id="run123",
        artifact_path="artifacts",
        tracking_uri="http://mlflow",
        relative_ckpt_path="ckpt.ckpt",
        ckpt_is_lightning=True,
        device="cpu",
    )
