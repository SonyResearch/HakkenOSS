"""Hydra compose + :class:`TrainKGEConfig` integration."""

from pathlib import Path

import pytest

from hakken_models.core.configs.train_kge import TrainKGEConfig
from hakken_models.core.utils import compose_config
from hakken_models.models.kge.lightning import build_lit_kge_val_metric_hub


def _configs_dir() -> Path:
    root = Path.cwd()
    d = root / "configs"
    if not d.is_dir():
        pytest.skip("Run from hakken-models package root (configs/ missing)")
    return d


def test_compose_train_kge_has_val_metric_hub_default() -> None:
    cfg = compose_config(config_name="train_kge", config_dir=_configs_dir())
    train_cfg = TrainKGEConfig.from_omegaconf(cfg)
    assert train_cfg.val_metric_hub.enabled is True
    assert train_cfg.val_metric_hub.bundles is None
    hub = build_lit_kge_val_metric_hub(train_cfg.val_metric_hub)
    assert hub is not None
    assert {b.name for b in hub.metric_bundles} == {"mean_rank"}


def test_compose_val_metric_hub_kge_v1_relation_multilabel_preset() -> None:
    cfg = compose_config(
        config_name="train_kge",
        config_dir=_configs_dir(),
        overrides=["val_metric_hub=kge_v1"],
    )
    train_cfg = TrainKGEConfig.from_omegaconf(cfg)
    hub = __import__(
        "hakken_models.models.kge.lightning", fromlist=["build_lit_kge_val_metric_hub"]
    ).build_lit_kge_val_metric_hub(train_cfg.val_metric_hub, num_relations=5)
    assert hub is not None
    names = {b.name for b in hub.metric_bundles}
    assert names == {
        "mean_rank",
        "relation_f1_micro",
        "relation_f1_macro",
        "relation_precision_micro",
        "relation_precision_macro",
        "relation_recall_micro",
        "relation_recall_macro",
    }
    assert hub.metric_bundles[1].metric_kwargs["num_labels"] == 5


def test_val_metric_hub_override_disabled() -> None:
    cfg = compose_config(
        config_name="train_kge",
        config_dir=_configs_dir(),
        overrides=["val_metric_hub.enabled=false"],
    )
    train_cfg = TrainKGEConfig.from_omegaconf(cfg)
    assert train_cfg.val_metric_hub.enabled is False
    assert build_lit_kge_val_metric_hub(train_cfg.val_metric_hub) is None


def test_compose_loss_kge_ranking_focal_uses_relation_base() -> None:
    cfg = compose_config(
        config_name="train_kge",
        config_dir=_configs_dir(),
        overrides=["loss=kge_ranking_focal"],
    )
    train_cfg = TrainKGEConfig.from_omegaconf(cfg)
    assert train_cfg.loss.name == "RankingRelationLoss"
    assert train_cfg.loss.kwargs["relation_loss_kwargs"]["reduction"] == "mean"
