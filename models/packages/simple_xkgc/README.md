# Hakken Explainer

A Python package for explaining KGE predictions using path-based explanations.

## Quick Start

```bash
# Install dependencies
uv sync

# Run explainer
make explain DATASET=pubtator3-v0.4.0
```

## Overview

Hakken Explainer generates explanations for knowledge graph completion predictions by finding and scoring paths between entities. It uses a modular architecture with:

- **Candidate Finders**: Find explanation paths using corpus-based or latent space methods
- **Scorers**: Evaluate paths using necessary/sufficient scoring methods
- **Rerankers**: Rank explanations by scores or unique pathways

## Documentation

📚 **[Full Documentation](https://your-docs-url)** (built with MkDocs)

For detailed documentation including API reference, configuration guides, and examples, see the [documentation site](https://your-docs-url) or build it locally:

```bash
make docs-serve
```

## Installation

```bash
uv sync
```

## Usage

```bash
# Generate explanations
make explain DATASET=pubtator3-v0.4.0

# Run benchmarks
make benchmark DATASET=pubtator3-v0.4.0

# Run checks
make run_checks
```

## Configuration

Configuration is managed via Hydra YAML files in `config/`. Key settings:
- `triple_to_probe`: Target triple to explain
- `candidate_finder`: Method for finding paths (corpus_path, latent_kge, latent_random)
- `explainer`: Explainer settings including GNN model configuration
- `run.score_type_list`: Scoring methods (necessary/sufficient)

## Contributing

1. Fork and create a feature branch
2. Make changes and run `make run_checks`
3. Submit a pull request