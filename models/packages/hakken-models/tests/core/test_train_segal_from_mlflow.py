"""TrainSeGALConfig from MLflow run params."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hakken_models.core.configs.train_segal import TrainSeGALConfig


def test_train_segal_config_from_mlflow_empty_params_raises() -> None:
    run = SimpleNamespace(data=SimpleNamespace(params={}))
    with patch("hakken_models.core.configs.train_common.mlflow.get_run", return_value=run):
        with pytest.raises(RuntimeError, match="No params"):
            TrainSeGALConfig.from_mlflow("abc")


def test_train_segal_config_from_mlflow_minimal_params() -> None:
    """Flat params sufficient for :class:`TrainSeGALConfig` (segal is required)."""
    run = SimpleNamespace(
        data=SimpleNamespace(
            params={
                "run/seed": "99",
                "segal/encoder_dim": "32",
                "segal/embedder/model_name": "m",
                "segal/embedder/embedding_dim": "32",
                "num_negatives": "8",
                "last_checkpoint_path": "/tmp/ignore.ckpt",
            }
        )
    )
    with patch("hakken_models.core.configs.train_common.mlflow.get_run", return_value=run):
        cfg = TrainSeGALConfig.from_mlflow("abc")

    assert cfg.run.seed == 99
    assert cfg.segal.encoder_dim == 32
    assert cfg.num_negatives == 8


def test_train_segal_config_from_mlflow_param_overrides() -> None:
    run = SimpleNamespace(
        data=SimpleNamespace(
            params={
                "run/seed": "99",
                "segal/encoder_dim": "32",
                "segal/embedder/model_name": "m",
                "segal/embedder/embedding_dim": "32",
                "num_negatives": "8",
            }
        )
    )
    with patch("hakken_models.core.configs.train_common.mlflow.get_run", return_value=run):
        cfg = TrainSeGALConfig.from_mlflow(
            "abc",
            param_overrides={"run/seed": "100"},
        )

    assert cfg.run.seed == 100
