# Candidate Finders

Candidate finders are responsible for identifying potential explanation paths between entities in the knowledge graph.

## Overview

A candidate finder takes a source entity, target entity, and optional constraints, then returns a list of paths that could explain the relationship between them.

## Available Finders

### CorpusPathFinder

Finds paths directly in the knowledge graph using NetworkX graph algorithms.

**When to use**: When you want explanations based on actual paths in the graph structure.

**Configuration**:
```yaml
_target_: hakken_explainer.candidate_finder.corpus.CorpusPathFinder
max_candidates: 50000
undirected: true
```

**Features**:
- Finds all simple paths up to a specified length
- Can filter by allowed relations
- Supports both directed and undirected graph traversal
- Efficient caching of graph structures

### LatentKGECandidateFinder

Uses Knowledge Graph Embedding (KGE) model embeddings to find paths in the latent space.

**When to use**: When you want to leverage learned embeddings to find semantically similar paths.

**Configuration**:
```yaml
_target_: hakken_explainer.candidate_finder.latent.kge.KGEPathCandidateFinder
max_candidates: 50000
path_generator:
  _target_: hakken_explainer.candidate_finder.path_generator.kge.KGEPathGenerator
  kge_model_path: ${oc.env:KGE_FOLDER}
```

**Features**:
- Uses embedding similarity to find paths
- Can discover paths not directly present in the graph
- Leverages learned semantic relationships

### LatentRandomCandidateFinder

Generates random paths in the latent space.

**When to use**: For baseline comparisons or when exploring the latent space.

**Configuration**:
```yaml
_target_: hakken_explainer.candidate_finder.latent.random.RandomPathCandidateFinder
max_candidates: 50000
path_generator:
  _target_: hakken_explainer.candidate_finder.path_generator.random.RandomPathGenerator
```

## Common Parameters

All candidate finders support:

- **`max_candidates`**: Maximum number of candidate paths to return
- **`undirected`**: Whether to treat the graph as undirected (default: `true`)

## Usage

```python
from hakken_explainer.candidate_finder.corpus.path import CorpusPathFinder

finder = CorpusPathFinder(max_candidates=1000)
finder.setup(facts_batch=search_space, kg=kg, kge=kge)

candidates = finder.find_candidates(
    source=subject_idx,
    target=object_idx,
    k=2,  # Path length
    allowed_relations=[relation_idx]  # Optional filter
)
```

## Custom Candidate Finders

To create a custom candidate finder, inherit from `CandidateFinder` and implement:

```python
from hakken_explainer.candidate_finder.base import CandidateFinder

class MyCandidateFinder(CandidateFinder):
    def find_candidates(
        self,
        source: int,
        target: int,
        relation: int | None = None,
        k: int | None = None,
        allowed_relations: list[int] | None = None,
    ) -> list[FactIndexList]:
        # Your implementation
        pass
```

See the [API Reference](../api/candidate-finder.md) for more details.

