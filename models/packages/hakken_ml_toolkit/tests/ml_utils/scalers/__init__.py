import importlib

import pytest

try:
    importlib.util.find_spec("torch")
except ImportError:
    pytest.skip("PyTorch is not installed", allow_module_level=True)
