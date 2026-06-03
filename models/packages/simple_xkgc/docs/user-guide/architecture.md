# Architecture

Hakken Explainer follows a modular architecture that separates concerns and allows for flexible configuration.

## Component Overview

```
┌─────────────────────────────────────────────────────────┐
│                    HakkenExplainer                      │
│  (Orchestrates the explanation process)                 │
└──────────────┬──────────────────────────────────────────┘
               │
    ┌──────────┴──────────┐
    │                     │
┌───▼──────────┐   ┌──────▼──────────┐   ┌──────────────┐
│  Candidate   │   │     Scorer      │   │  Reranker    │
│   Finder     │──▶│                 │──▶│              │
└──────────────┘   └─────────────────┘   └──────────────┘
```

## Core Components

### HakkenExplainer

The main entry point that coordinates the explanation process. It:

- Manages the candidate finder, model, and knowledge graph
- Orchestrates the finding, scoring, and reranking steps
- Returns a DataFrame with ranked explanations

### Candidate Finder

Responsible for finding potential explanation paths. Implementations include:

- **CorpusPathFinder**: Finds paths directly in the knowledge graph using NetworkX
- **LatentKGECandidateFinder**: Uses KGE embeddings to find paths in latent space
- **LatentRandomCandidateFinder**: Generates random paths in latent space

All candidate finders inherit from `CandidateFinder` and implement:
- `setup()`: Initialize with knowledge graph facts
- `find_candidates()`: Find paths between source and target entities

### Scorer

Evaluates how well explanation paths support predictions. Implementations:

- **SufficientScore**: Measures if a path alone justifies the prediction
- **NecessaryScore**: Measures if a path is required for the prediction

All scorers inherit from `ExplainerScore` and implement:
- `score()`: Compute scores for candidate paths

### Reranker

Orders explanations by relevance. Implementations:

- **ScoreReranker**: Ranks by average score
- **UniquePathwayReranker**: Ranks by unique pathways, then by score

## Data Flow

1. **Input**: Triple to explain `(subject, relation, object)`
2. **Candidate Finding**: Find paths from subject to object
3. **Scoring**: Evaluate each path's support for the prediction
4. **Reranking**: Order explanations by relevance
5. **Output**: DataFrame with ranked explanations and scores

## Extension Points

The architecture is designed for extensibility:

- **New Candidate Finders**: Implement `CandidateFinder` interface
- **New Scoring Methods**: Implement `ExplainerScore` interface
- **New Rerankers**: Implement `ExplanationReranker` interface

See the [API Reference](../api/explainers.md) for details on extending the system.

