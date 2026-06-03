"""TrainKGEConfig from MLflow run params."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import typer

from hakken_models.core.configs.train_kge import TrainKGEConfig
from hakken_models.core.utils.runtime import flat_overrides_from_override_list, parse_override


def test_train_kge_config_from_mlflow_empty_params_raises() -> None:
    run = SimpleNamespace(data=SimpleNamespace(params={}))
    with patch("hakken_models.core.configs.train_common.mlflow.get_run", return_value=run):
        with pytest.raises(RuntimeError, match="No params"):
            TrainKGEConfig.from_mlflow("abc")


def test_train_kge_config_from_mlflow_minimal_params() -> None:
    run = SimpleNamespace(
        data=SimpleNamespace(
            params={
                "run/seed": "99",
                "last_checkpoint_path": "/tmp/ignore.ckpt",
            }
        )
    )
    with patch("hakken_models.core.configs.train_common.mlflow.get_run", return_value=run):
        cfg = TrainKGEConfig.from_mlflow("abc")

    assert cfg.run.seed == 99


def test_train_kge_config_from_mlflow_param_overrides() -> None:
    run = SimpleNamespace(data=SimpleNamespace(params={"run/seed": "99"}))
    with patch("hakken_models.core.configs.train_common.mlflow.get_run", return_value=run):
        cfg = TrainKGEConfig.from_mlflow(
            "abc",
            param_overrides={"run/seed": "100"},
        )

    assert cfg.run.seed == 100


def test_flat_overrides_from_override_list() -> None:
    assert flat_overrides_from_override_list(None) == {}
    assert flat_overrides_from_override_list([]) == {}
    assert flat_overrides_from_override_list(["run/seed=7"]) == {"run/seed": "7"}
    assert flat_overrides_from_override_list(parse_override("run/seed=1 kge/name=RotatE")) == {
        "run/seed": "1",
        "kge/name": "RotatE",
    }


def test_flat_overrides_from_override_list_rejects_bad_token() -> None:
    with pytest.raises(typer.BadParameter):
        flat_overrides_from_override_list(["noseparator"])
