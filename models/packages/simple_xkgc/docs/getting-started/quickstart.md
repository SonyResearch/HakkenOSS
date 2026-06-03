# Quick Start

This guide will help you get started with Hakken Explainer in minutes.

## Basic Usage

The simplest way to use Hakken Explainer is through the command line:

```bash
make explain DATASET=pubtator3-v0.4.0
```

This will:
1. Load the configuration for the specified dataset
2. Find candidate explanation paths
3. Score the paths using the configured scoring method
4. Output explanations to `explanations.tsv`

## Programmatic Usage

You can also use Hakken Explainer programmatically:

```python
from hakken_explainer.explainers import HakkenExplainer
from hakken_explainer.candidate_finder.corpus import CorpusPathFinder
from hakken_explainer.entities.config import ScoreTypeConfig
from hakken_explainer.constants import ScoreType

# Initialize candidate finder
candidate_finder = CorpusPathFinder(max_candidates=1000)
candidate_finder.setup(facts_batch=search_space, kg=kg, kge=kge)

# Initialize explainer
explainer = HakkenExplainer(
    candidate_finder=candidate_finder,
    model=gnn_model,
    kg=kg,
    search_space=search_space
)

# Generate explanations
score_configs = [ScoreTypeConfig(type=ScoreType.SUFFICIENT, batch_size=32)]
explanations = explainer.explain(
    triple_to_probe=("entity1", "relation", "entity2"),
    device="cuda",
    explanation_length=2,
    score_type_list=score_configs
)
```

## Configuration

Hakken Explainer uses Hydra for configuration management. The main configuration file is `config/config.yaml`:

```yaml
defaults:
  - triple_to_probe: pubtator3-v0.4.0
  - candidate_finder: corpus_path
  - explainer: default

run:
  device: "cuda"
  score_type_list:
    - _target_: hakken_explainer.entities.config.ScoreTypeConfig
      type: sufficient
      batch_size: 32
```

See the [Configuration](configuration.md) page for detailed information.

## Output Format

The explainer outputs a TSV file with the following columns:

- `explanation`: Human-readable explanation path
- `pathway`: List of entity pairs in the path
- `score`: Average score across all scoring methods
- `score_sufficient`: Sufficient score (if computed)
- `score_necessary`: Necessary score (if computed)

## Next Steps

- Learn about [Candidate Finders](user-guide/candidate-finders.md)
- Understand [Scoring Methods](user-guide/scoring-methods.md)
- Explore [Advanced Configuration](examples/advanced-config.md)

