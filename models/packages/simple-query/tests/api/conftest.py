from pathlib import Path

import pytest


@pytest.fixture
def envfile_path() -> Path:
    return Path(__file__).parent / "test_config.env"


@pytest.fixture
def yaml_path() -> Path:
    return Path(__file__).parent / "test_config.yaml"
