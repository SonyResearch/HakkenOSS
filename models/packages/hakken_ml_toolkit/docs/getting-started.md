# Getting Started



## Installation

This package uses UV for dependency management. To install the package, follow these steps:

Ensure you have UV installed. If not, install it by following the instructions at https://docs.astral.sh/uv/getting-started/installation/


Install the package and its dependencies:
  ```bash
  uv sync --index-strategy unsafe-best-match
  ```


## Running Tests

```bash
uv run pytest
```

## Explore Demos

Open the Jupyter notebooks in `notebooks/`:

* `file_manager_demo.ipynb`
* `losses_demo.ipynb`
* `optimizers_demo.ipynb`
