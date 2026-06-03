from pathlib import Path

import typer
from loguru import logger
from zenml.integrations.mlflow.flavors import MLFlowExperimentTrackerConfig

from hakken_models.core.configs.dataset_preparator import DatasetPreparatorConfig
from hakken_models.core.configs.evaluate_model import EvaluateModelConfig
from hakken_models.core.configs.train_kge import TrainKGEConfig
from hakken_models.core.configs.train_segal import TrainSeGALConfig
from hakken_models.core.configs.train_thiger import TrainTHiGERConfig
from hakken_models.core.utils import compose_config
from hakken_models.core.utils.runtime import flat_overrides_from_override_list, parse_override
from hakken_models.pipelines.dataset_preparation import dataset_preparation_pipeline
from hakken_models.pipelines.kge_training import train_kge_pipeline
from hakken_models.pipelines.kge_tuning import tune_kge
from hakken_models.pipelines.segal_evaluation import evaluate_segal_pipeline
from hakken_models.pipelines.segal_training import train_segal_pipeline
from hakken_models.pipelines.thiger_evaluation import evaluate_thiger_pipeline
from hakken_models.pipelines.thiger_training import train_thiger_pipeline
from hakken_models.pipelines.thiger_tuning import tune_thiger

app = typer.Typer(help="Hakken models pipeline CLI")


@app.command()
def prepare_dataset(
    config_dir: Path | None = typer.Option(  # noqa: B008
        None, "--config-dir", help="Directory containings configs (defaults to 'configs')"
    ),
    override: str | None = typer.Option(
        None,
        help=(
            "Override configuration values using key=value pairs "
            "(e.g., 'kge=rotate' or 'kge=rotate optimizer=adamw'). "
            "Multiple overrides can be space-separated."
        ),
    ),
    print_config: bool = typer.Option(
        False, "--print-config", help="Print the final config and exit"
    ),
):
    """
    Prepare the dataset by running the dataset preparation pipeline.
    """
    # Format S3 paths if using default templates

    if override is not None:
        override = override.split(" ")

    cfg = compose_config(config_name="prepare_dataset", config_dir=config_dir, overrides=override)
    config = DatasetPreparatorConfig.from_omegaconf(cfg.dataset_preparator)
    if print_config:
        typer.echo(config.model_dump_json(indent=2))
        return

    dataset_preparation_pipeline.with_options(
        tags=[config.dataset_name, config.dataset_version], enable_cache=False
    )(config=config)


@app.command()
def train_kge(
    config_dir: Path | None = typer.Option(  # noqa: B008
        None, "--config-dir", help="Directory containings configs (defaults to 'configs')"
    ),
    from_mlflow_run: str | None = typer.Option(
        None,
        "--from-mlflow-run",
        help=(
            "If set, build the training config from this MLflow run's logged params "
            "(instead of Hydra compose). Use MLFLOW_TRACKING_URI or --mlflow-tracking-uri."
        ),
    ),
    mlflow_tracking_uri: str | None = typer.Option(
        None,
        "--mlflow-tracking-uri",
        help="MLflow tracking URI when using --from-mlflow-run (optional if env is set).",
    ),
    from_yaml: Path | None = typer.Option(
        None,
        "--from-yaml",
        help=(
            "Path to a local YAML file with a full TrainKGE config tree (e.g. exported params). "
            "Mutually exclusive with --from-mlflow-run; when set, Hydra compose is not used."
        ),
    ),
    override: str | None = typer.Option(
        None,
        help=(
            "Override configuration values using key=value pairs "
            "(e.g., 'kge=rotate' or 'kge=rotate optimizer=adamw'). "
            "Multiple overrides can be space-separated. "
            "With --from-mlflow-run, overrides are merged as flat keys on top of run params "
            "(same key style as logged params, e.g. run/seed=1). "
            "With --from-yaml, overrides use dotted Hydra-style keys (e.g. run.seed=1)."
        ),
    ),
    print_config: bool = typer.Option(
        False, "--print-config", help="Print the final config and exit"
    ),
):
    """
    Train the model using the specified dataset and model name.
    """

    override_list = parse_override(override)

    if from_mlflow_run is not None and from_yaml is not None:
        raise typer.BadParameter("Use only one of --from-mlflow-run or --from-yaml")

    if from_yaml is not None:
        config = TrainKGEConfig.from_file(from_yaml, overrides=override_list)
    elif from_mlflow_run is not None:
        run_id = from_mlflow_run.strip()
        if not run_id:
            raise typer.BadParameter("--from-mlflow-run must be a non-empty run id")
        flat_overrides = flat_overrides_from_override_list(override_list)
        config = TrainKGEConfig.from_mlflow(
            run_id,
            tracking_uri=mlflow_tracking_uri,
            param_overrides=flat_overrides or None,
        )
    else:
        cfg = compose_config(
            config_name="train_kge", config_dir=config_dir, overrides=override_list
        )
        config = TrainKGEConfig.from_omegaconf(cfg)
    if print_config:
        typer.echo(config.model_dump_json(indent=2))
        return

    mlflow_settings = MLFlowExperimentTrackerConfig(
        experiment_name=config.experiment_tracker.experiment_name
    )
    train_kge_pipeline.with_options(
        settings={"experiment_tracker.mlflow": mlflow_settings},
        run_name=config.experiment_tracker.run_name,
    )(config=config)


@app.command()
def train_segal(
    config_dir: Path | None = typer.Option(  # noqa: B008
        None, "--config-dir", help="Directory containing configs (defaults to 'configs')"
    ),
    from_mlflow_run: str | None = typer.Option(
        None,
        "--from-mlflow-run",
        help=(
            "If set, build the training config from this MLflow run's logged params "
            "(instead of Hydra compose). Use MLFLOW_TRACKING_URI or --mlflow-tracking-uri."
        ),
    ),
    mlflow_tracking_uri: str | None = typer.Option(
        None,
        "--mlflow-tracking-uri",
        help="MLflow tracking URI when using --from-mlflow-run (optional if env is set).",
    ),
    from_yaml: Path | None = typer.Option(
        None,
        "--from-yaml",
        help=(
            "Path to a local YAML file with a full TrainSeGAL config tree (e.g. exported params). "
            "Mutually exclusive with --from-mlflow-run; when set, Hydra compose is not used."
        ),
    ),
    override: str | None = typer.Option(
        None,
        help=(
            "Override configuration values using key=value pairs "
            "(e.g., 'segal.gnn.name=GAT optimizer=adamw'). "
            "Multiple overrides can be space-separated. "
            "With --from-mlflow-run, overrides are merged as flat keys on top of run params "
            "(same key style as logged params, e.g. run/seed=1). "
            "With --from-yaml, overrides use dotted Hydra-style keys (e.g. run.seed=1)."
        ),
    ),
    print_config: bool = typer.Option(
        False, "--print-config", help="Print the final config and exit"
    ),
):
    """
    Train a SeGAL model using pre-computed embeddings and temporal context.
    """

    override_list = parse_override(override)

    if from_mlflow_run is not None and from_yaml is not None:
        raise typer.BadParameter("Use only one of --from-mlflow-run or --from-yaml")

    if from_yaml is not None:
        config = TrainSeGALConfig.from_file(from_yaml, overrides=override_list)
    elif from_mlflow_run is not None:
        run_id = from_mlflow_run.strip()
        if not run_id:
            raise typer.BadParameter("--from-mlflow-run must be a non-empty run id")
        flat_overrides = flat_overrides_from_override_list(override_list)
        config = TrainSeGALConfig.from_mlflow(
            run_id,
            tracking_uri=mlflow_tracking_uri,
            param_overrides=flat_overrides or None,
        )
    else:
        cfg = compose_config(
            config_name="train_segal", config_dir=config_dir, overrides=override_list
        )
        config = TrainSeGALConfig.from_omegaconf(cfg)
    if print_config:
        typer.echo(config.model_dump_json(indent=2))
        return

    mlflow_settings = MLFlowExperimentTrackerConfig(
        experiment_name=config.experiment_tracker.experiment_name
    )
    train_segal_pipeline.with_options(
        settings={"experiment_tracker.mlflow": mlflow_settings},
        run_name=config.experiment_tracker.run_name,
    )(config=config)


@app.command()
def hpo_kge(
    config_dir: Path | None = typer.Option(  # noqa: B008
        None, "--config-dir", help="Directory containings configs (defaults to 'configs')"
    ),
    override: str | None = typer.Option(
        None,
        help=(
            "Override configuration values using key=value pairs "
            "(e.g., 'kge=rotate' or 'kge=rotate optimizer=adamw'). "
            "Multiple overrides can be space-separated."
        ),
    ),
    print_config: bool = typer.Option(
        False, "--print-config", help="Print the final config and exit"
    ),
):
    """
    Train the model using the specified dataset and model name.
    """

    from hakken_models.hpo.build_tuner import build_tuner
    from hakken_models.hpo.search_space.kge import optuna_search_space

    override_list = None
    if override is not None:
        override_list = override.split(" ")

    cfg = compose_config(config_name="train_kge", config_dir=config_dir, overrides=override_list)
    config = TrainKGEConfig.from_omegaconf(cfg)
    if print_config:
        typer.echo(config.model_dump_json(indent=2))
        return

    tuner = build_tuner(
        config=config.hpo,
        trainable=tune_kge,
        search_space=optuna_search_space,
        override_str=override,
        config_dir=config_dir,
    )

    results = tuner.fit()
    if config.hpo is None:
        raise RuntimeError("hpo_kge requires a non-null hpo config (e.g. override hpo=default)")
    metric_name = config.hpo.metric.name
    metric_mode = config.hpo.metric.mode
    best_result = results.get_best_result(metric=metric_name, mode=metric_mode)
    logger.info(f"Best trial config: {best_result.config}")
    logger.info(f"Best {metric_name}: {best_result.metrics[metric_name]}")


@app.command()
def evaluate_kge(
    config_dir: Path | None = typer.Option(  # noqa: B008
        None, "--config-dir", help="Directory containing configs (defaults to 'configs')"
    ),
    override: str | None = typer.Option(
        None,
        help=(
            "Override configuration values using key=value pairs "
            "(e.g., 'thiger.gnn=graphsage'). "
            "Multiple overrides can be space-separated."
        ),
    ),
    print_config: bool = typer.Option(
        False, "--print-config", help="Print the final config and exit"
    ),
):
    """
    Evaluate KGE model using the specified dataset and model name.
    """
    from hakken_models.pipelines.kge_evaluation import evaluate_kge_pipeline

    if override is not None:
        override = override.split(" ")

    cfg = compose_config(config_name="evaluate_kge", config_dir=config_dir, overrides=override)
    logger.info(cfg)
    config = EvaluateModelConfig.from_omegaconf(cfg)
    if print_config:
        typer.echo(config.model_dump_json(indent=2))
        return

    mlflow_settings = MLFlowExperimentTrackerConfig(
        experiment_name=config.experiment_tracker.experiment_name
    )

    evaluate_kge_pipeline.with_options(settings={"experiment_tracker.mlflow": mlflow_settings})(
        config=config
    )


@app.command()
def train_thiger(
    config_dir: Path | None = typer.Option(  # noqa: B008
        None, "--config-dir", help="Directory containing configs (defaults to 'configs')"
    ),
    override: str | None = typer.Option(
        None,
        help=(
            "Override configuration values using key=value pairs "
            "(e.g., 'thiger.gnn=graphsage'). "
            "Multiple overrides can be space-separated."
        ),
    ),
    print_config: bool = typer.Option(
        False, "--print-config", help="Print the final config and exit"
    ),
):
    """
    Train the THiGER using the specified dataset and model name.
    """

    override_list = parse_override(override)

    cfg = compose_config(config_name="train_thiger", config_dir=config_dir, overrides=override_list)

    config = TrainTHiGERConfig.from_omegaconf(cfg)

    if print_config:
        typer.echo(config.model_dump_json(indent=2))
        return

    mlflow_settings = MLFlowExperimentTrackerConfig(
        experiment_name=config.experiment_tracker.experiment_name
    )

    train_thiger_pipeline.with_options(
        settings={"experiment_tracker.mlflow": mlflow_settings},
        run_name=config.experiment_tracker.run_name,
    )(config=config)


@app.command()
def hpo_thiger(
    config_dir: Path | None = typer.Option(  # noqa: B008
        None, "--config-dir", help="Directory containings configs (defaults to 'configs')"
    ),
    override: str | None = typer.Option(
        None,
        help=(
            "Override configuration values using key=value pairs "
            "(e.g., 'kge=rotate' or 'kge=rotate optimizer=adamw'). "
            "Multiple overrides can be space-separated."
        ),
    ),
    print_config: bool = typer.Option(
        False, "--print-config", help="Print the final config and exit"
    ),
):
    """
    Train the model using the specified dataset and model name.
    """

    from hakken_models.hpo.build_tuner import build_tuner
    from hakken_models.hpo.search_space.thiger import optuna_search_space

    override_list = None
    if override is not None:
        override_list = override.split(" ")

    cfg = compose_config(config_name="train_thiger", config_dir=config_dir, overrides=override_list)
    config = TrainTHiGERConfig.from_omegaconf(cfg)
    if print_config:
        typer.echo(config.model_dump_json(indent=2))
        return

    tuner = build_tuner(
        config=config.hpo,
        trainable=tune_thiger,
        search_space=optuna_search_space,
        override_str=override,
        config_dir=config_dir,
    )

    results = tuner.fit()
    best_result = results.get_best_result(metric="val_macro_avg_f1_score", mode="max")
    logger.info(f"Best trial config: {best_result.config}")
    logger.info(f"Best val_macro_avg_f1_score: {best_result.metrics['val_macro_avg_f1_score']}")


@app.command()
def evaluate_thiger(
    config_dir: Path | None = typer.Option(  # noqa: B008
        None, "--config-dir", help="Directory containing configs (defaults to 'configs')"
    ),
    override: str | None = typer.Option(
        None,
        help=(
            "Override configuration values using key=value pairs "
            "(e.g., 'thiger.gnn=graphsage'). "
            "Multiple overrides can be space-separated."
        ),
    ),
    print_config: bool = typer.Option(
        False, "--print-config", help="Print the final config and exit"
    ),
):
    """
    Train the THiGER using the specified dataset and model name.
    """
    if override is not None:
        override = override.split(" ")

    cfg = compose_config(config_name="evaluate_thiger", config_dir=config_dir, overrides=override)
    logger.info(cfg)
    config = EvaluateModelConfig.from_omegaconf(cfg)
    if print_config:
        typer.echo(config.model_dump_json(indent=2))
        return

    mlflow_settings = MLFlowExperimentTrackerConfig(
        experiment_name=config.experiment_tracker.experiment_name
    )

    evaluate_thiger_pipeline.with_options(settings={"experiment_tracker.mlflow": mlflow_settings})(
        config=config
    )


@app.command()
def evaluate_segal(
    config_dir: Path | None = typer.Option(  # noqa: B008
        None, "--config-dir", help="Directory containing configs (defaults to 'configs')"
    ),
    override: str | None = typer.Option(
        None,
        help=(
            "Override configuration values using key=value pairs "
            "(e.g., 'model_loader.mlflow_run_id=abc123'). "
            "Multiple overrides can be space-separated."
        ),
    ),
    print_config: bool = typer.Option(
        False, "--print-config", help="Print the final config and exit"
    ),
):
    """
    Evaluate a trained SeGAL model on the specified dataset splits.
    """
    if override is not None:
        override = override.split(" ")

    cfg = compose_config(config_name="evaluate_segal", config_dir=config_dir, overrides=override)
    logger.info(cfg)
    config = EvaluateModelConfig.from_omegaconf(cfg)
    if print_config:
        typer.echo(config.model_dump_json(indent=2))
        return

    mlflow_settings = MLFlowExperimentTrackerConfig(
        experiment_name=config.experiment_tracker.experiment_name
    )

    evaluate_segal_pipeline.with_options(settings={"experiment_tracker.mlflow": mlflow_settings})(
        config=config
    )


if __name__ == "__main__":
    app()
