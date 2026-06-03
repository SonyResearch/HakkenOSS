# Hakken ML Toolkit

A collection of ML utilities and base structures used across Hakken packages.

## Overview

Provides reusable components for training and evaluating ML models:

- **metrics** — evaluation metric implementations
- **losses** — custom loss functions
- **optimizers** — optimizer utilities
- **ml_base_structures** — base classes and interfaces for models
- **ml_utils** — general-purpose ML helper utilities
- **tracker** — experiment tracking helpers
- **file_manager** — model checkpoint and file management

## Installation

```bash
uv sync
```

## Usage

```python
from hakken_ml_toolkit.metrics import ...
from hakken_ml_toolkit.losses import ...
```

## Testing

```bash
uv run pytest
```
