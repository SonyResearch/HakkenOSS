
#  Knowledge Graph Embeddings (KGE) Package

A comprehensive PyTorch-based framework for knowledge graph embeddings.


## Overview

This framework provides tools for training, evaluating, and using various Knowledge Graph Embedding (KGE) models. It features:

- Multiple KGE Models: ComplEx, DistMult, ConvE, and GNN-based models
- PyTorch Lightning Integration
- Negative Sampling Strategies: Configurable negative sampling for training
- Comprehensive Evaluation: Multiple metrics for relation and entity prediction
- Filtration Mechanisms: Hard filtering for fair model evaluation
- Experiment Tracking: Integration with ML tracking tools
- GNN Extensions: Train Graph Neural Networks on top of trained KGE models
- Hydra Configuration: Flexible experiment configuration

## Installation


Create the `.env`:
```bash
cp .env.example .env
```
Update the variables in this file. You have to load your env vars by `source .env`.

Install dependencies:

```bash
uv sync
```

To check everything is working run the tests with the following command:

```bash
uv run pytest
```

## ID vs Index
The framework distinguishes between two types of identifiers:

- **ID**: Human-readable string identifiers for entities and relations (e.g., "person_123", "works_for")
- **Index**: Numerical indices used internally by the model for tensor operations (e.g., 0, 1, 2...)


## Usage

See `misc/demo.ipynb` for an example on training and inference on ComplEx.


### Training and Inference
```bash
# Training
make train DATASET=digital_science MODEL=complex

# Inference
make inference

# Training a GNN to mimic a KGE
uv run python kge/delivery/cli/train_mimic_kge.py  data_repo=digital_science trained_kge=digital_science model=sagekge 
```

To check the training progress, you can use ML Flow:

```bash
make run_mlflow
```

### Run Hyperparameter Optimization

```bash
make run_hpo DATASET=digital_science MODEL=complex PROJECT_HPO=ds-complex
```

### Experimentation

Use the `misc/` directory for experimental notebooks.



## Project Structure

```
.
├── README.md                   # Project documentation
├── config/                     # Configuration files
│   ├── inference/              # Inference configuration
│   └── training/               # Training configuration
├── kge/                        # Main package
│   ├── common/                 # Common utilities and domain models
│   │   ├── actions/            # Action classes for various operations
│   │   ├── constants.py        # Constant definitions
│   │   ├── data_generator.py   # Data generation utilities
│   │   ├── domain.py           # Domain model definitions
│   │   └── exceptions.py       # Custom exceptions
│   ├── data_processing/        # Data loading and processing
│   ├── delivery/               # Command-line interfaces
│   │   └── cli/                # CLI implementations
│   ├── early_stopping/         # Early stopping strategies
│   ├── evaluator/              # Metrics and evaluation utilities
│   ├── models/                 # KGE model implementations
│   │   ├── base.py             # Base model interfaces
│   │   ├── complex.py          # ComplEx model
│   │   ├── conv_e/             # ConvE model
│   │   ├── distmult.py         # DistMult model
│   │   ├── er_model.py         # Entity-Relation model base
│   │   ├── gnn.py              # GNN-based models
│   │   ├── kge_api.py          # KGE API for external access
│   │   └── mlp.py              # MLP-based models
│   ├── negative_sampler/       # Negative sampling strategies
│   ├── optim/                  # Optimization utilities
│   ├── scores/                 # Scoring functions
│   ├── trainer/                # Training utilities
│   │   └── lightning/          # PyTorch Lightning modules
│   └── triple_filterer/        # Filtering utilities for evaluation
├── misc/                       # Miscellaneous utilities
├── scripts/                    # Utility scripts
└── tests/                      # Test suite
```

## Core Components

### Models

The framework implements several KGE models:

- ComplEx: Complex embeddings for simple link prediction
- DistMult: A simplified bilinear model
- ConvE: Convolutional embeddings for link prediction
- GNN-based models: Models that combine KGE with graph neural networks

### Data

- Flexible data loaders for different input formats
- Utilities for handling knowledge graph triples
- Support for different data splits (train, validation, test)

### Training

- PyTorch Lightning integration for efficient training
- Support for various optimizers and learning rate schedulers
- Early stopping strategies for improved convergence

### Evaluation

- Comprehensive ranking metrics
- Hard filtering mechanism for fair evaluation
- Support for multiple prediction targets (subject, relation, object)



### Extending the framework

1. Create a new model class in `kge/models/`
2. Inherit from `KGEI` (or `ERModel`) base class
3. Implement required methods
4. Add a configuration class that inherits from `KGEConfig`
5. Add tests in `tests/kge`



## Configuration

The framework uses Hydra for configuration management. Key configuration files inside `config/` folder.



## Model Registry

| Model Version | Dataset   | Dataset Version | Package Version |
|:------------|:----------|---------------:|---------------:|
| kge-v1.0.0       | Pubtator3 | 0.3.0          | 0.1.7  (approx)    |
| kge-v1.1.0       | Pubtator3 | 0.4.0          | 0.2.52       |
| xkge-v1.1.0       | Pubtator3 | 0.4.0         | 0.2.52       |
| kge-v1.2.0       | DigitalScience | 2.0.0     | 0.1.8 (appox)  |



## Troubleshooting

if you encounter problems when running `mlflow ui` try finding the process running on port 5000 and kill it:

```bash
sudo lsof -i :5000
sudo kill <PID>
```


