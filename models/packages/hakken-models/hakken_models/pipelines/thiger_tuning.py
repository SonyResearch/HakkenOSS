def tune_thiger(config: dict, override_str: str | None, config_dir: str | None) -> None:
    from zenml.integrations.mlflow.flavors import MLFlowExperimentTrackerConfig

    from hakken_models.core.configs.train_thiger import TrainTHiGERConfig
    from hakken_models.core.utils import compose_config
    from hakken_models.pipelines.thiger_training import train_thiger_pipeline

    override_list = []
    if override_str is not None:
        override_list = override_str.split(" ")

    trial_overrides = [f"{k}={v}" for k, v in config.items()]

    cfg = compose_config(
        config_name="train_thiger",
        config_dir=config_dir,
        overrides=override_list + trial_overrides,
    )
    train_config = TrainTHiGERConfig.from_omegaconf(cfg)
    if train_config.hpo is None:
        raise ValueError("HPO config must be provided for tuning")

    mlflow_settings = MLFlowExperimentTrackerConfig(
        experiment_name=train_config.experiment_tracker.experiment_name
    )
    train_thiger_pipeline.with_options(
        settings={"experiment_tracker.mlflow": mlflow_settings},
        run_name=train_config.experiment_tracker.run_name,
    )(config=train_config)
