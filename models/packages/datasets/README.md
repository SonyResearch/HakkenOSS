# Datasets

This repository contains a Python package for processing and managing knowledge graph datasets..

## Overview

This library provides a standardized interface for working with knowledge graph datasets in machine learning workflows. It offers:

- Unified data loading
- Data caching
- Format conversion.  Tools to transform different source formats into a common representation
- Dataset utilities to split, filter, and process knowledge graph data
- PyTorch integration. Ready-to-use data loaders for PyTorch models.

### Supported datasets
The package currently supports:

1. DigitalScience
2. TextKG -  A general text-based knowledge graph loader supporting various temporal and static datasets:
   - ICEWS14 ([source](https://github.com/mniepert/mmkb/tree/master/TemporalKGs))
   - Countries/Nations ([source](https://github.com/ZhenfengLei/KGDatasets/tree/master/Nations))

Custom datasets can be added by implementing the `DataRepositoryI` interface.



### Project Structure

```bash
.
├── datasets
│   ├── __init__.py                # Package exports
│   ├── common                     # Common utilities
│   │   ├── constants.py           # Enum definitions
│   │   ├── domain.py              # Type definitions
│   │   └── exceptions.py          # Custom exceptions
│   ├── data_loader_manager.py     # DataLoader utilities
│   └── data_repo                  # Dataset implementations
│       ├── base.py                # Base repository interface
│       ├── digital_science        # DigitalScience dataset
│       └── text.py                # Text-based KG datasets
└── tests                          # Unit tests
```
## Installation

This package uses UV for dependency management. To install the package, follow these steps:

Ensure you have UV installed. If not, install it by following the instructions at https://docs.astral.sh/uv/getting-started/installation/


Install the package and its dependencies:
  ```bash
  uv sync
  ```

## Usage

The file `misc/demo.ipynb` contains some basic examples of how to use the classes provided by this package. 

## Running Tests

To run the tests, use the following command:

```bash
uv run pytest
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.