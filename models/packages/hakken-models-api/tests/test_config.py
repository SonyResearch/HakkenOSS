"""Tests for get_config() and HakkenModelsAPIConfig."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from hakken_models.core.constants import ModelType
from hakken_models_api.config import HakkenModelsAPIConfig, get_config

CONFIG_ENV_KEYS = (
    "MODEL",
    "DEVICE",
    "CKPT_IS_LIGHTNING",
    "RELATIVE_CKPT_PATH",
    "RELATIVE_PARAMS_PATH",
    "RUN_DIR",
    "MLFLOW_RUN_ID",
    "ARTIFACT_PATH",
    "TRACKING_URI",
    "DATA_ROOT_URI_TEMPLATE",
    "ENV_FILE",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove config-relevant env vars so tests are isolated."""
    for key in CONFIG_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------------------
# HakkenModelsAPIConfig — direct construction
# ---------------------------------------------------------------------------


class TestHakkenModelsAPIConfig:
    def test_dir_mode(self):
        cfg = HakkenModelsAPIConfig(run_dir="s3://bucket/model/")
        assert cfg.use_local_dir is True
        assert cfg.use_mlflow is False
        assert cfg.model == ModelType.THIGER

    def test_mlflow_mode(self):
        cfg = HakkenModelsAPIConfig(mlflow_run_id="abc123")
        assert cfg.use_mlflow is True
        assert cfg.use_local_dir is False

    def test_both_modes_raises(self):
        with pytest.raises(ValidationError, match="mutually exclusive"):
            HakkenModelsAPIConfig(run_dir="/path", mlflow_run_id="abc123")

    def test_neither_mode_raises(self):
        with pytest.raises(ValidationError, match="must be provided"):
            HakkenModelsAPIConfig()

    def test_model_type_segal(self):
        cfg = HakkenModelsAPIConfig(model="segal", run_dir="/path")
        assert cfg.model == ModelType.SEGAL

    def test_empty_run_dir_coerced_to_none(self):
        cfg = HakkenModelsAPIConfig(run_dir="", mlflow_run_id="abc123")
        assert cfg.run_dir is None
        assert cfg.use_mlflow is True

    def test_empty_mlflow_run_id_coerced_to_none(self):
        cfg = HakkenModelsAPIConfig(mlflow_run_id="", run_dir="/path")
        assert cfg.mlflow_run_id is None
        assert cfg.use_local_dir is True

    def test_whitespace_only_coerced_to_none(self):
        cfg = HakkenModelsAPIConfig(run_dir="   ", mlflow_run_id="abc123")
        assert cfg.run_dir is None

    def test_defaults(self):
        cfg = HakkenModelsAPIConfig(run_dir="/path")
        assert cfg.device == "cuda"
        assert cfg.ckpt_is_lightning is True
        assert cfg.relative_ckpt_path == "last/last.ckpt"
        assert cfg.relative_params_path == "params.json"
        assert cfg.artifact_path == "checkpoints"
        assert cfg.model == ModelType.THIGER


# ---------------------------------------------------------------------------
# get_config() — env-var driven
# ---------------------------------------------------------------------------


class TestGetConfig:
    @pytest.fixture(autouse=True)
    def _isolate_cwd(self, tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
        """Run each test in a temp dir so stray .env files don't leak."""
        monkeypatch.chdir(tmp_path)

    def test_dir_mode_from_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("RUN_DIR", "s3://bucket/model/")
        cfg = get_config()
        assert cfg.run_dir == "s3://bucket/model/"
        assert cfg.use_local_dir is True

    def test_mlflow_mode_from_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("MLFLOW_RUN_ID", "run789")
        monkeypatch.setenv("TRACKING_URI", "http://localhost:5000")
        cfg = get_config()
        assert cfg.mlflow_run_id == "run789"
        assert cfg.tracking_uri == "http://localhost:5000"
        assert cfg.use_mlflow is True

    def test_model_type_from_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("MODEL", "segal")
        monkeypatch.setenv("RUN_DIR", "/some/path")
        cfg = get_config()
        assert cfg.model == ModelType.SEGAL

    def test_env_file_override(self, monkeypatch: pytest.MonkeyPatch, tmp_path):
        env_file = tmp_path / ".env.test"
        env_file.write_text("RUN_DIR=s3://test-bucket/\nMODEL=segal\n")
        monkeypatch.setenv("ENV_FILE", str(env_file))
        cfg = get_config()
        assert cfg.run_dir == "s3://test-bucket/"
        assert cfg.model == ModelType.SEGAL

    def test_empty_run_dir_env_uses_mlflow(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("RUN_DIR", "")
        monkeypatch.setenv("MLFLOW_RUN_ID", "abc123")
        cfg = get_config()
        assert cfg.run_dir is None
        assert cfg.use_mlflow is True

    def test_neither_mode_from_env_raises(self):
        with pytest.raises(ValidationError, match="must be provided"):
            get_config()
