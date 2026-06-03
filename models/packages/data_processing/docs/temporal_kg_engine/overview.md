# Temporal Knowledge Graph Engine

## Overview

The Temporal Knowledge Graph Engine (`data_processing.temporal_kg_engine`) is a framework for ingesting, managing, and querying temporal knowledge graphs. It supports multiple graph database backends (InMemory) via a unified interface.

## Purpose

- Ingests large-scale temporal data efficiently
- Manages domain-typed nodes and relation-typed edges
- Supports temporal queries (year-based filtering, path finding)
- Provides a consistent API across different backends

## Architecture

The engine follows an abstract base class pattern:

```
TemporalKGEngine (Abstract Base Class)
└── InMemoryTKGEngine (InMemory Implementation)
```

### Key Components

- **Base Engine** (`TemporalKGEngine`): Defines the common interface and shared logic
- **Implementation Engines**: Backend-specific implementations
    - `InMemoryTKGEngine`: In-memoery based using Networkx and polars
- **Settings Classes**: Configuration via Pydantic (environment variables)
- **Factory Pattern**: Engine creation from environment or settings

## Key Features

### Data Ingestion
- Batch processing for large datasets
- TSV file support (nodes and edges)
- Automatic data validation
- Index creation for query performance

### Temporal Queries
- Year-based filtering
- Shortest path finding (single and all paths)
- Temporal relationship traversal
- Property-based filtering

### Multi-Backend Support
- Unified API across implementations
- Easy switching between backends
- Backend-specific optimizations

### Operations
- Database cleanup and recreation
- Import verification
- Result limiting
- Connection management

## Usage Example

```python
from data_processing.temporal_kg_engine.factory import TKGFactory

# Create engine from environment
engine = TKGFactory.from_env("in_memory")  

# Connect and ingest
engine.connect()
stats = engine.ingest(
    nodes_file_path="nodes.tsv",
    edges_file_path="edges.tsv",
    recreate=True
)

# Query temporal relationships
paths = engine.find_shortest_path(
    source_id="node_1",
    target_id="node_2",
    max_depth=10
)
```

## Data Model

- **Nodes**: Typed by domain (e.g., `Person`, `Organization`, `Disease`)
- **Edges**: Typed by relation (e.g., `KNOWS`, `WORKS_FOR`, `INHIBIT`)
- **Properties**: Node and edge properties including temporal attributes
- **Year**: Edges include `year` for temporal filtering

## Configuration

Configure via environment variables or Pydantic settings:

```bash
# Common settings
GRAPH_NAME=temporal_kg
BATCH_SIZE=10000
```

## Requirements

- Python 3.11+
- Docker & Docker Compose
- Graph database backend (or non if using in memory)
- Dependencies: `polars`, `neo4j`, `loguru`, `pydantic`

## Installation

```bash
uv sync --extra docs
```

Configure your `.env` file and start your chosen backend service with Docker Compose.

## Design Principles

1. **Abstraction**: Single interface for multiple backends
2. **Performance**: Batch processing and indexing
3. **Flexibility**: Environment-based configuration
4. **Temporal Focus**: Built-in support for time-based queries
5. **Type Safety**: Pydantic settings and type hints

## Contributing

To add a new backend implementation:
1. Extend `TemporalKGEngine`
2. Implement all abstract methods
3. Create a corresponding settings class
4. Update the factory pattern
5. Add integration tests

---

For detailed setup and operations, see the implementation-specific documentation.