"""Tests for SeGALRunLoader."""

from unittest.mock import MagicMock

import pytest

from hakken_models.models.segal import SeGALArtifacts

from hakken_models_api.config import HakkenModelsAPIConfig
from hakken_models_api.loaders.segal import SeGALRunLoader
from spaice_inference_api import ModelLoadingOptions


def test_segal_run_loader_load_from_dir(monkeypatch: pytest.MonkeyPatch):
    """Test that SeGALRunLoader delegates to SeGALLoader.load()."""
    fake_artifacts = MagicMock(spec=SeGALArtifacts)

    def mock_load(_self=None):
        return fake_artifacts

    monkeypatch.setattr(
        "hakken_models_api.loaders.segal.SeGALLoader",
        MagicMock(return_value=MagicMock(load=mock_load)),
    )

    config = HakkenModelsAPIConfig(
        model="segal",
        run_dir="s3://bucket/models/segal-v1/",
        relative_params_path="params.json",
        relative_ckpt_path="last.ckpt",
        ckpt_is_lightning=True,
        device="cpu",
    )

    logger = MagicMock()
    loader = SeGALRunLoader()

    result = loader.load(
        options=ModelLoadingOptions(path=""),
        logger=logger,
        config=config,
    )

    assert result is fake_artifacts
