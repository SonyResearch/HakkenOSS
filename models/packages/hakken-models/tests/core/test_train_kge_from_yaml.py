"""TrainKGEConfig from a local YAML file."""

from pathlib import Path

import pytest

from hakken_models.core.configs.train_kge import TrainKGEConfig


def _write_minimal_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text(
        """
experiment_tracker:
  experiment_name: exp-from-yaml
run:
  seed: 7
dataset:
  name: d
  version: v1
  data_root_uri_template: s3://bucket/{name}/{version}
  load_embeddings: false
data_loader:
  name: DataLoader
  kwargs: {}
optimizer:
  name: Adam
  kwargs: {lr: 0.01}
kge:
  embedding_dim: 64
  score_fn_name: ComplExScore
negative_sampler:
  name: UniformNegativeSampler
  kwargs:
    corruption_scheme: [subject, object]
negative_strategy:
  name: mean
logger:
  log_model: false
model_checkpoint:
  monitor: val/mean_rank
  filename: best
  every_n_epochs: 1
  checkpoint_dir: checkpoints
  save_last: true
  mode: min
  save_top_k: 1
  enabled: false
init_strategy:
  name: XavierNormal
  kwargs: {}
trainer:
  max_epochs: 1
  devices: 1
  strategy: auto
  check_val_every_n_epoch: 1
  auto_batch_size: false
  gradient_clip_val: null
loss:
  name: MarginRankingLoss
  kwargs: {}
val_metric_hub:
  enabled: false
  bundles: []
""".strip()
    )
    return p


def test_train_kge_config_from_yaml_minimal(tmp_path: Path) -> None:
    p = _write_minimal_yaml(tmp_path)
    cfg = TrainKGEConfig.from_file(p)
    assert cfg.experiment_tracker.experiment_name == "exp-from-yaml"
    assert cfg.run.seed == 7


def test_train_kge_config_from_yaml_override(tmp_path: Path) -> None:
    p = _write_minimal_yaml(tmp_path)
    cfg = TrainKGEConfig.from_file(p, overrides=["run.seed=42"])
    assert cfg.run.seed == 42


def test_train_kge_config_from_yaml_missing_file() -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        TrainKGEConfig.from_file("/nonexistent/path/to/cfg.yaml")
