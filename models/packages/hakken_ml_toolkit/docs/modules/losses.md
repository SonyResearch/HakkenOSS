# Losses

Losses is a Python package that provides a collection of customizable loss functions for PyTorch-based machine learning models, with a focus on classification, regression, and ranking tasks. The library currently includes:

- Binary Cross Entropy with Logits Loss
- Mean Squared Error Loss
- Margin Ranking Loss

## Project Structure

```bash
.
├── losses/
│   ├── __init__.py                # Package exports
│   ├── base/                      # Abstract base classes
│   │   ├── __init__.py
│   │   ├── clf_loss.py            # Classification loss base
│   │   ├── ranking_loss.py        # Ranking loss base
│   │   └── regression_loss.py     # Regression loss base
│   ├── common/                    # Shared utilities
│   │   ├── __init__.py
│   │   ├── constants.py           # Enums and constants
│   │   ├── domain.py              # Type definitions
│   │   └── exceptions.py          # Custom exceptions
│   ├── bce_with_logits_loss.py    # BCE implementation
│   ├── margin_ranking.py          # Margin ranking implementation
│   ├── mse.py                     # MSE implementation
│   └── py.typed                   # For mypy
└── tests/                         # Test suite
    ├── __init__.py
    ├── test_bce_with_logits.py
    └── test_margin_ranking.py
```
