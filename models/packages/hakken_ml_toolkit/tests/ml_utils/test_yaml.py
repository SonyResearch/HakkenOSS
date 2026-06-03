from pathlib import Path

import pytest
from omegaconf import DictConfig

from hakken_ml_toolkit.ml_utils import YAMLUtils
from hakken_ml_toolkit.ml_utils.exceptions import InvalidYamlError


@pytest.fixture
def sample_yaml_content() -> str:
    return """
    key1: value1
    key2:
      nested_key: nested_value
    key3:
      - list_item1
      - list_item2
    """


@pytest.fixture
def sample_yaml_file(tmp_path: Path, sample_yaml_content: str) -> Path:
    file_path = tmp_path / "sample.yaml"
    file_path.write_text(sample_yaml_content)
    return file_path


@pytest.fixture
def multiple_yaml_files(tmp_path: Path) -> list[Path]:
    content1 = "key1: value1\nkey2: value2"
    content2 = "key3: value3\nkey2: new_value2"

    file1 = tmp_path / "file1.yaml"
    file2 = tmp_path / "file2.yaml"

    file1.write_text(content1)
    file2.write_text(content2)

    return [file1, file2]


def test_load_yaml(sample_yaml_file: Path) -> None:
    result = YAMLUtils.load(sample_yaml_file)

    assert isinstance(result, dict)
    assert result["key1"] == "value1"
    assert result["key2"]["nested_key"] == "nested_value"
    assert result["key3"] == ["list_item1", "list_item2"]


def test_load_yaml_with_string_path(sample_yaml_file: Path) -> None:
    result = YAMLUtils.load(str(sample_yaml_file))

    assert isinstance(result, dict)
    assert result["key1"] == "value1"


def test_load_nonexistent_file() -> None:
    with pytest.raises(FileNotFoundError):
        YAMLUtils.load("nonexistent.yaml")


def test_load_many_yaml(multiple_yaml_files: list[Path]) -> None:
    result = YAMLUtils.load_many([str(f) for f in multiple_yaml_files])

    assert isinstance(result, DictConfig)
    assert result.key1 == "value1"
    assert result.key2 == "new_value2"  # Second file overwrites key2
    assert result.key3 == "value3"


def test_load_many_empty_list() -> None:
    with pytest.raises(InvalidYamlError):
        YAMLUtils.load_many([])


def test_load_many_invalid_yaml() -> None:
    invalid_content = "invalid: : yaml"
    file_path = Path("invalid.yaml")
    file_path.write_text(invalid_content)

    with pytest.raises(InvalidYamlError):
        YAMLUtils.load_many([str(file_path)])

    file_path.unlink()  # Clean up the temporary file


def test_load_many_single_file(sample_yaml_file: Path) -> None:
    result = YAMLUtils.load_many([str(sample_yaml_file)])

    assert isinstance(result, DictConfig)
    assert result.key1 == "value1"
    assert result.key2.nested_key == "nested_value"
    assert result.key3 == ["list_item1", "list_item2"]
