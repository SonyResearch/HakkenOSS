# Graph neural Networks Package

A flexible and extensible Graph Neural Network library built on PyTorch Geometric.


## Features

- **Multiple GNN Implementations**: Includes GCN, GAT, GIN, RGCN, and GraphSAGE with a unified API
- **Modular Architecture**: Separate pre-processing, graph convolution, and post-processing stages
- **Flexible Configuration**: Comprehensive configuration options for all models
- **Graph/Node Level Predictions**: Support for both node-level and graph-level tasks
- **Skip Connections**: Built-in support for residual-style skip connections
- **Pooling Operations**: Various pooling strategies for graph-level tasks



## Folder Structure

```
.
├── gnn/                           # Main package directory
│   ├── __init__.py                # Package exports
│   ├── architectures/             # GNN model implementations
│   │   ├── __init__.py
│   │   ├── base.py                # Abstract base classes
│   │   ├── gat.py                 # Graph Attention Network
│   │   ├── gcn.py                 # Graph Convolutional Network
│   │   ├── gin.py                 # Graph Isomorphism Network
│   │   ├── graphsage.py           # GraphSAGE implementation
│   │   └── rgcn.py                # Relational GCN
│   ├── common/                    # Shared utilities
│   │   ├── __init__.py
│   │   ├── constants.py           # Enums and constants
│   │   ├── domain.py              # Type definitions
│   │   └── exceptions.py          # Custom exceptions
│   ├── data.py                    # Data preparation utilities
│   ├── mlp.py                     # Multi-layer perceptron
│   ├── node_wrapper.py            # Node feature wrapper
│   └── pooling.py                 # Graph pooling operations
├── playground/                    # Example notebooks
└── tests/                         # Unit tests
```

## Installation

```bash
uv sync --index-strategy unsafe-best-match
```

## Usage

The file `misc/demo.ipynb` contains some basic examples of how to use the classes provided by this package. 


## Running Tests

To run the tests, use the following command:

```bash
uv run pytest
```

## Architecture Details

The library uses a three-stage architecture for all GNN models:

1. **Pre-processing Stage**: An MLP that transforms input node features
2. **Graph Stage**: GNN-specific layers that perform message passing
3. **Post-processing Stage**: An MLP that transforms node embeddings after message passing
4. **Pooling** (optional): Aggregates node embeddings to graph-level representations

This modular approach allows for flexible combinations of different components.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.



