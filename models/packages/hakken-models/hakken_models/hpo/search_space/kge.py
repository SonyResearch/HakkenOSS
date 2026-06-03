from collections.abc import Callable
from typing import Any

import optuna


def add_param(config: dict, key: str, func: Callable, *args, **kwargs) -> None:
    config[key] = func(key, *args, **kwargs)


def optuna_search_space(trial: optuna.Trial) -> dict[str, Any]:
    """
    Define-by-run search space for KGE.
    """
    config: dict[str, Any] = {}

    add_param(config, "optimizer.kwargs.lr", trial.suggest_float, 1e-4, 5e-2, log=True)

    add_param(
        config,
        "kge",
        trial.suggest_categorical,
        ["rotate", "complex"],
    )

    add_param(
        config,
        "kge.embedding_dim",
        trial.suggest_categorical,
        [128, 256, 512],
    )

    if config["kge"] == "rotate":
        add_param(
            config,
            "kge.score_fn_kwargs.epsilon",
            trial.suggest_float,
            1.0,
            5.0,
        )

    add_param(
        config,
        "negative_strategy.name",
        trial.suggest_categorical,
        ["mean"],
    )

    add_param(
        config,
        "data_loader.kwargs.num_negatives",
        trial.suggest_categorical,
        [2, 10, 50, 100],
    )

    add_param(
        config,
        "loss",
        trial.suggest_categorical,
        ["kge_ranking_focal", "kge_ranking_bce"],
    )

    add_param(
        config,
        "loss.kwargs.entity_loss_kwargs.margin",
        trial.suggest_float,
        1.0,
        20.0,
    )

    add_param(
        config,
        "loss.kwargs.rel_loss_weight",
        trial.suggest_categorical,
        [0.0, 0.25, 0.5, 1.0],
    )

    if config["loss"] == "kge_ranking_focal":
        add_param(
            config,
            "loss.kwargs.relation_loss_kwargs.gamma",
            trial.suggest_float,
            0.5,
            5.0,
        )

        add_param(
            config,
            "loss.kwargs.relation_loss_kwargs.regularization_coeff",
            trial.suggest_categorical,
            [0.0, 0.25, 0.5, 1.0],
        )

    return config
