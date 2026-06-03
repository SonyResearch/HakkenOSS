# Rerankers

Rerankers order explanations by relevance and quality after scoring.

## Overview

After candidate paths are scored, rerankers determine the final ordering of explanations. Different reranking strategies prioritize different aspects of the explanations.

## Available Rerankers

### ScoreReranker

Ranks explanations by their average score across all scoring methods.

**When to use**: When you want the highest-scoring explanations first.

**Strategy**: `RerankStrategy.SCORES`

**Behavior**:
- Sorts by average score (descending)
- Preserves all explanations

### PathwayReranker

Ranks by unique pathways first, then by score within each pathway group.

**When to use**: When you want diverse explanations that cover different paths.

**Strategy**: `RerankStrategy.UNIQUE_PATHWAYS`

**Behavior**:
- Groups explanations by unique pathways
- Ranks pathways by their best score
- Within each pathway, ranks by score

## Usage

Rerankers are automatically applied by `HakkenExplainer`, but you can specify the strategy:

```python
from hakken_explainer.constants import RerankStrategy

explanations = explainer.explain(
    triple_to_probe=triple,
    rerank_strategy=RerankStrategy.UNIQUE_PATHWAYS
)
```

## Configuration

The reranking strategy can be set in the explainer configuration or passed directly to the `explain()` method. By default, `RerankStrategy.SCORES` is used.

## Custom Rerankers

To create a custom reranker, inherit from `ExplanationReranker`:

```python
from hakken_explainer.reranker.base import ExplanationReranker
import pandas as pd

class MyReranker(ExplanationReranker):
    def rerank(self, explanations_df: pd.DataFrame) -> pd.DataFrame:
        # Your reranking logic
        return explanations_df.sort_values(by='score', ascending=False)
```

See the [API Reference](../api/reranker.md) for more details.

