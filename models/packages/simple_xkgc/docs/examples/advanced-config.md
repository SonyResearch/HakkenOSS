# Advanced Configuration

## Custom Candidate Finder Configuration

Create a custom candidate finder configuration:

```yaml
# config/candidate_finder/custom.yaml
_target_: hakken_explainer.candidate_finder.corpus.path.CorpusPathCandidateFinder
max_candidates: 10000
undirected: false  # Use directed graph
```

Use it:
```bash
make explain candidate_finder=custom
```

## Multiple Scoring Methods with Different Batch Sizes

Optimize batch sizes for different scoring methods:

```yaml
run:
  score_type_list:
    - _target_: hakken_explainer.entities.config.ScoreTypeConfig
      type: sufficient
      batch_size: 64  # Larger batch for sufficient
    - _target_: hakken_explainer.entities.config.ScoreTypeConfig
      type: necessary
      batch_size: 4   # Smaller batch for necessary (memory intensive)
```

## Environment Variable Substitution

Use environment variables in configuration:

```yaml
# config/explainer/custom.yaml
_target_: hakken_explainer.entities.config.HakkenExplainerConfig
data_path: ${oc.env:DATA_ROOT_FOLDER}
graph_cache_folder: ${oc.env:GRAPH_CACHE_FOLDER}/my_cache
gnn_experiment_config:
  experiment_folder: ${oc.env:MODEL_FOLDER}
```

## Custom Output Format

Configure output location and format:

```yaml
output:
  path: results/explanations_${now:%Y%m%d_%H%M%S}.tsv
  delimiter: ","
```

## Relation Filtering

Filter paths by allowed relations:

```yaml
run:
  relation_filter: ["relation_1", "relation_2"]
```

Or via command line:
```bash
make explain run.relation_filter='["relation_1","relation_2"]'
```

## Device Configuration

Switch between devices:

```yaml
run:
  device: "cpu"  # or "cuda"
```

## Logging Configuration

Control log verbosity:

```yaml
log_level: DEBUG  # DEBUG, INFO, WARNING, ERROR
```

## Combining Multiple Overrides

Override multiple settings at once:

```bash
make explain \
  DATASET=pubtator3-v0.4.0 \
  candidate_finder=latent_kge \
  run.device=cpu \
  run.score_type_list='[{_target_: hakken_explainer.entities.config.ScoreTypeConfig, type: sufficient, batch_size: 16}]' \
  output.path=custom_output.tsv
```

## Programmatic Configuration

Create configurations programmatically:

```python
from omegaconf import DictConfig, OmegaConf
from hakken_explainer.explainers import HakkenExplainer

# Create config
cfg = OmegaConf.create({
    "candidate_finder": {
        "_target_": "hakken_explainer.candidate_finder.corpus.path.CorpusPathCandidateFinder",
        "max_candidates": 5000
    },
    "run": {
        "device": "cuda",
        "score_type_list": [...]
    }
})

# Use with explainer
candidate_finder = hydra.utils.instantiate(cfg.candidate_finder)
# ... rest of setup
```

