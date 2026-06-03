from pathlib import Path

import pytest

from hakken_ml_toolkit.ml_utils import StringUtils


@pytest.fixture
def sample_string() -> str:
    return "Hello, World!"


def test_to_txt_write_mode(sample_string: str, tmp_path: Path) -> None:
    file_path = tmp_path / "test.txt"
    result = StringUtils.to_txt_file(sample_string, str(file_path))

    assert result == str(file_path)
    assert file_path.exists()

    with open(file_path) as f:
        content = f.read()

    assert content == "Hello, World!\n"


def test_to_txt_append_mode(sample_string: str, tmp_path: Path) -> None:
    file_path = tmp_path / "test.txt"

    # Write initial content
    StringUtils.to_txt_file("Initial content\n", str(file_path))

    # Append new content
    result = StringUtils.to_txt_file(sample_string, str(file_path), mode="a")

    assert result == str(file_path)

    with open(file_path) as f:
        content = f.read()

    assert content == "Initial content\nHello, World!\n"


def test_to_txt_newline_handling(tmp_path: Path) -> None:
    file_path = tmp_path / "test.txt"

    # Test with string that already has a newline
    StringUtils.to_txt_file("Hello\n", str(file_path))

    with open(file_path) as f:
        content = f.read()

    assert content == "Hello\n"
    assert len(content) == 6  # Ensuring only one newline is present


def test_to_txt_empty_string(tmp_path: Path) -> None:
    file_path = tmp_path / "test.txt"
    with pytest.raises(IndexError):
        _ = StringUtils.to_txt_file("", str(file_path))

    assert not file_path.exists()
