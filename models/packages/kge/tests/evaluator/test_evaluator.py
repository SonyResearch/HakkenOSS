import os
from typing import cast

import pytest
import torch
import yaml
from dotenv import load_dotenv
from hakken_ml_toolkit.ml_base_structures.data_generator import DummyDataGenerator
from hydra.utils import instantiate

from kge.evaluator import KGEEvaluator, KGEEvaluatorConfig

load_dotenv()


def default_config() -> dict:
    config_folder = os.getenv("CONFIG_PATH", "config")

    evaluator_config_yaml = os.path.join(config_folder, "evaluator", "default.yaml")

    with open(evaluator_config_yaml) as file:
        cfg = cast("dict", yaml.safe_load(file))
    return cast("dict", cfg["config"])


def test_load_evaluator_config():
    """
    Test that the evaluator config can be loaded.
    """
    cfg = default_config()
    assert isinstance(cfg, dict)
    assert "_target_" in cfg


@pytest.mark.parametrize("seed", [42, 123, 456])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_init(seed: int, device: str):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    # Arrange
    cfg = default_config()
    config = cast("KGEEvaluatorConfig", instantiate(cfg))
    assert isinstance(config, KGEEvaluatorConfig)

    evaluator = KGEEvaluator(config)

    # Sanity: before init, nothing should be set
    assert evaluator._model is None
    assert evaluator.kg is None
    assert getattr(evaluator, "ranking_metrics", None) in (None, [])
    assert getattr(evaluator, "relation_clf_metrics", None) in (None, [])
    assert getattr(evaluator, "targets", None) in (None, set())

    kg = DummyDataGenerator.knowledge_graph_from_seed(seed=seed, device=device)

    # Act
    evaluator.init(kg)

    # Assert: core state
    assert evaluator.kg is kg
    assert evaluator.ranking_metrics, "ranking_metrics should be created"
    assert evaluator.relation_clf_metrics, "relation_clf_metrics should be created"
    assert evaluator.targets, "targets should be populated from hakken_ml_toolkit.metrics config"

    # Assert: filterers exist for all targets when filter_list is set
    assert set(evaluator.filterer.keys()) == set(evaluator.targets)
