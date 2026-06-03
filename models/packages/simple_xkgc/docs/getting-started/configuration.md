# Configuration

Hakken Explainer uses [Hydra](https://hydra.cc/) for configuration management, allowing flexible and composable configuration through YAML files.

## Configuration Structure

The configuration is organized in the `config/` directory:

```
config/
├── config.yaml              # Main configuration
├── candidate_finder/        # Candidate finder configurations
│   ├── corpus_path.yaml
│   ├── latent_kge.yaml
│   └── latent_random.yaml
├── explainer/               # Explainer configurations
│   └── default.yaml
└── triple_to_probe/         # Dataset-specific triple configurations
    ├── pubtator3-v0.4.0.yaml
    └── digital_science-v.2.0.0.yaml
```

## Main Configuration

The `config.yaml` file defines the overall settings:

```yaml
defaults:
  - _self_
  - triple_to_probe: pubtator3-v0.4.0
  - candidate_finder: corpus_path
  - explainer: default

log_level: INFO

output:
  path: explanations.tsv
  delimiter: "\t"

run:
  device: "cuda"
  score_type_list:
    - _target_: hakken_explainer.entities.config.ScoreTypeConfig
      type: sufficient
      batch_size: 32
  relation_filter: null
```

### Key Settings

- **`triple_to_probe`**: Which dataset/triple configuration to use
- **`candidate_finder`**: Method for finding explanation paths
- **`explainer`**: Explainer configuration (model, data paths, etc.)
- **`run.device`**: Device for computation (`cuda` or `cpu`)
- **`run.score_type_list`**: List of scoring methods to apply
- **`output.path`**: Output file path

## Candidate Finder Configuration

Choose from different candidate finding strategies:

### Corpus Path Finder

Finds paths directly in the knowledge graph:

```yaml
_target_: hakken_explainer.candidate_finder.corpus.path.CorpusPathCandidateFinder
max_candidates: 50000
undirected: true
```

### Latent KGE Finder

Uses KGE embeddings to find paths in latent space:

```yaml
_target_: hakken_explainer.candidate_finder.latent.kge.LatentKGECandidateFinder
max_candidates: 50000
path_generator:
  _target_: hakken_explainer.candidate_finder.path_generator.kge.KGEPathGenerator
  kge_model_path: ${oc.env:KGE_FOLDER}
```

## Explainer Configuration

The explainer configuration specifies model and data paths:

```yaml
_target_: hakken_explainer.entities.config.HakkenExplainerConfig
data_path: ${oc.env:DATA_ROOT_FOLDER}
search_space_split_names: [train, val, test]
graph_cache_folder: ${oc.env:GRAPH_CACHE_FOLDER}
gnn_experiment_config:
  _target_: kge.common.entities.kge_loader_config.KGELoadExperimentConfig
  experiment_folder: ${oc.env:MODEL_FOLDER}
  config_path: .hydra/config.yaml
  model_ckpt_path: seed_0/model_checkpoint/last.ckpt
  model_ckpt_is_lightning: True
  device: ${run.device}
```

## Environment Variables

Configuration can reference environment variables using `${oc.env:VAR_NAME}`:

- `DATA_ROOT_FOLDER`: Base directory for datasets
- `MODEL_FOLDER`: Path to trained GNN model
- `GRAPH_CACHE_FOLDER`: Directory for caching graphs
- `KGE_FOLDER`: Path to trained KGE model
- `CONFIG_PATH`: Path to configuration directory

## Overriding Configuration

You can override configuration values via command line:

```bash
make explain DATASET=pubtator3-v0.4.0 \
  candidate_finder=latent_kge \
  run.device=cpu \
  output.path=my_explanations.tsv
```

## Advanced Configuration

For more advanced configuration options, see the [Advanced Configuration](examples/advanced-config.md) examples.

