# Scoring Methods

Scoring methods evaluate how well explanation paths support knowledge graph completion predictions.

## Overview

Each scoring method answers a different question about the explanation:

- **Sufficient**: "Is this path alone enough to justify the prediction?"
- **Necessary**: "Is this path required for the prediction?"

## Sufficient Scoring

Measures whether an explanation path alone is sufficient to justify the prediction.

**Concept**: If we only had this path as context, would the model still make the same prediction?

**Use case**: Finding minimal explanations that independently support the prediction.

**Implementation**: `SufficientScore`

```python
from hakken_explainer.scores.sufficient import SufficientScore
from hakken_explainer.entities.config import ScoreTypeConfig
from hakken_explainer.constants import ScoreType

scorer = SufficientScore(context_kg=search_space, model=gnn_model)
scores = scorer.score(
    target_fact=target_triple,
    candidate_paths=candidate_paths,
    batch_size=32,
    num_hops=2,
    normalize_by_original=True
)
```

**Configuration**:
```yaml
- _target_: hakken_explainer.entities.config.ScoreTypeConfig
  type: sufficient
  batch_size: 32
```

## Necessary Scoring

Measures whether an explanation path is necessary for the prediction.

**Concept**: If we remove this path from the context, would the model still make the same prediction?

**Use case**: Finding critical paths that the model depends on.

**Implementation**: `NecessaryScore`

```python
from hakken_explainer.scores.necessary import NecessaryScore

scorer = NecessaryScore(context_kg=search_space, model=gnn_model)
scores = scorer.score(
    target_fact=target_triple,
    candidate_paths=candidate_paths,
    batch_size=8,
    num_hops=2
)
```

**Configuration**:
```yaml
- _target_: hakken_explainer.entities.config.ScoreTypeConfig
  type: necessary
  batch_size: 8
```

## Scoring Parameters

### batch_size

Controls how many paths are scored in parallel. Larger batches are faster but use more memory.

- **Sufficient scoring**: Typically uses larger batches (32-64)
- **Necessary scoring**: Typically uses smaller batches (8-16) due to higher memory requirements

### num_hops

Number of hops to include in the subgraph context when scoring. Default is 2.

### normalize_by_original

Whether to normalize scores by the original prediction score. When `True`, scores are relative to the model's original prediction.

## Using Multiple Scoring Methods

You can compute both sufficient and necessary scores:

```yaml
run:
  score_type_list:
    - _target_: hakken_explainer.entities.config.ScoreTypeConfig
      type: sufficient
      batch_size: 32
    - _target_: hakken_explainer.entities.config.ScoreTypeConfig
      type: necessary
      batch_size: 8
```

The output DataFrame will contain both `score_sufficient` and `score_necessary` columns, plus an averaged `score` column.

## Custom Scoring Methods

To create a custom scorer, inherit from `ExplainerScore`:

```python
from hakken_explainer.scores.base import ExplainerScore

class MyScorer(ExplainerScore):
    def score(
        self,
        target_fact: Tensor,
        candidate_paths: Tensor,
        batch_size: int = 1,
        num_hops: int = 2,
        normalize_by_original: bool = False,
    ) -> list[float]:
        # Your implementation
        pass
```

See the [API Reference](../api/scores.md) for more details.

