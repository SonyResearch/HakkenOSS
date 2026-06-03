import os
from collections.abc import Callable

from optuna.samplers import TPESampler
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

from hakken_models.core.configs.hpo import HPOConfig


def build_tuner(config: HPOConfig, trainable, search_space: Callable, **kwargs) -> tune.Tuner:
    scheduler = ASHAScheduler(
        time_attr="training_iteration",  # or "epoch" if you report per epoch
        max_t=20,  # max epochs
        grace_period=10,  # min epochs before a trial can be stopped
        reduction_factor=3,
    )

    search_alg = OptunaSearch(
        search_space,
        metric=config.metric.name,
        mode=config.metric.mode,
        sampler=TPESampler(seed=42),
    )

    return tune.Tuner(
        # Wrap the trainable so it accepts extra fixed args if needed
        tune.with_resources(
            tune.with_parameters(trainable=trainable, **kwargs),
            resources=config.resources_per_trial,
        ),
        tune_config=tune.TuneConfig(
            metric=config.metric.name,
            mode=config.metric.mode,
            num_samples=config.num_trials,
            scheduler=scheduler,
            search_alg=search_alg,
            trial_dirname_creator=None,
        ),
        run_config=tune.RunConfig(
            name=config.name,
            storage_path="file://" + os.path.abspath(config.relative_storage_path),
            verbose=config.verbose,
        ),
    )
