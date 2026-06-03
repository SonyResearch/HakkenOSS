# Hakken Explainer

A Python package for explaining Knowledge Graph Embedding (KGE) predictions using path-based explanations.

## Overview

Hakken Explainer generates explanations for knowledge graph completion predictions by finding and scoring paths between entities. The package provides a modular architecture that supports different methods for finding explanation paths and evaluating their quality.

## Key Features

- **Multiple Candidate Finding Strategies**: Find explanation paths using corpus-based methods or latent space exploration
- **Flexible Scoring**: Evaluate paths using necessary or sufficient scoring methods
- **Configurable Reranking**: Rank explanations by scores or unique pathways
- **GNN Integration**: Works seamlessly with Graph Neural Network models for knowledge graph completion
- **Performance Optimized**: Supports graph caching and efficient batch processing

## Quick Start

```bash
# Install dependencies
uv sync

# Run explainer
make explain DATASET=pubtator3-v0.4.0
```

## Documentation Structure

- **[Getting Started](getting-started/installation.md)**: Installation and setup instructions
- **[User Guide](user-guide/overview.md)**: Comprehensive guide to using Hakken Explainer
- **[Examples](examples/basic-usage.md)**: Code examples and use cases
- **[API Reference](api/explainers.md)**: Complete API documentation

## Architecture

Hakken Explainer follows a modular design with three main components:

1. **Candidate Finders**: Identify potential explanation paths between entities
2. **Scorers**: Evaluate how well paths support predictions
3. **Rerankers**: Order explanations by relevance

See the [Architecture](user-guide/architecture.md) page for detailed information.

