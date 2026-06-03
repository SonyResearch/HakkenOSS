from collections.abc import Callable
from typing import Any

import optuna


def add_param(config: dict, key: str, func: Callable, *args, **kwargs) -> None:
    config[key] = func(key, *args, **kwargs)


def optuna_search_space(trial: optuna.Trial) -> dict[str, Any]:
    """
    Define-by-run search space with conditionals.
    """
    config = {}

    add_param(
        config,
        "optimizer",
        trial.suggest_categorical,
        ["adam", "adamw"],
    )

    # Optimizer
    add_param(config, "optimizer.kwargs.lr", trial.suggest_float, 1e-4, 5e-2, log=True)

    # THiGER
    add_param(
        config,
        "thiger.entity_embedding_dim",
        trial.suggest_categorical,
        [32, 64, 128, 256, 512],
    )
    config["thiger.relation_embedding_dim"] = config["thiger.entity_embedding_dim"]

    add_param(
        config,
        "thiger.domain_embedding_dim",
        trial.suggest_categorical,
        [32, 64, 128],
    )

    add_param(
        config,
        "thiger.gnn.kwargs.hidden_channels",
        trial.suggest_categorical,
        [32, 64, 128],
    )

    add_param(
        config,
        "thiger.gnn.kwargs.dropout",
        trial.suggest_categorical,
        [0.0, 0.1, 0.2, 0.3],
    )

    add_param(
        config,
        "thiger.gnn.kwargs.act",
        trial.suggest_categorical,
        ["relu"],
    )

    add_param(
        config,
        "thiger.gnn.kwargs.norm",
        trial.suggest_categorical,
        ["batch", "layer", None],
    )

    add_param(
        config,
        "thiger.gnn.kwargs.jk",
        trial.suggest_categorical,
        ["last", "max", "cat", None],
    )

    # Data loader
    add_param(
        config,
        "data_loader.kwargs.num_neighbors",
        trial.suggest_categorical,
        [[64, 64], [128, 128], [512, 512]],
    )

    # Loss function
    add_param(
        config,
        "loss",
        trial.suggest_categorical,
        ["bce_with_logits", "focal"],
    )

    add_param(
        config,
        "loss.kwargs.margin",
        trial.suggest_float,
        1.0,
        100.0,
    )

    if config["loss"] == "focal":
        add_param(
            config,
            "loss.kwargs.gamma",
            trial.suggest_float,
            0.5,
            5.0,
        )

    return config
