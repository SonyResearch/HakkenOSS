from pathlib import Path
from typing import Any

import pytest
import yaml

from hakken_ml_toolkit.ml_utils import DictUtils


@pytest.fixture
def sample_dict() -> dict[str, Any]:
    return {"a": 1, "b": {"c": 2, "d": {"e": 3}}, "f": 4}


def test_flatten(sample_dict: dict[str, Any]) -> None:
    flattened = DictUtils.flatten(sample_dict)
    expected = {"a": 1, "b.c": 2, "b.d.e": 3, "f": 4}
    assert flattened == expected


def test_flatten_with_custom_delimiter(sample_dict: dict[str, Any]) -> None:
    flattened = DictUtils.flatten(sample_dict, delimiter="-")
    expected = {"a": 1, "b-c": 2, "b-d-e": 3, "f": 4}
    assert flattened == expected


def test_to_txt(sample_dict: dict[str, Any], tmp_path: Path) -> None:
    file_path = tmp_path / "test.txt"
    result = DictUtils.to_txt(sample_dict, str(file_path))
    assert result == str(file_path)
    assert file_path.exists()

    with open(file_path) as f:
        content = f.read()

    assert "a\t1\n" in content
    assert "f\t4\n" in content


def test_to_yaml(sample_dict: dict[str, Any], tmp_path: Path) -> None:
    file_path = tmp_path / "test.yaml"
    result = DictUtils.to_yaml(sample_dict, str(file_path))
    assert result == str(file_path)
    assert file_path.exists()

    with open(file_path) as f:
        loaded_dict = yaml.safe_load(f)

    assert loaded_dict == sample_dict
