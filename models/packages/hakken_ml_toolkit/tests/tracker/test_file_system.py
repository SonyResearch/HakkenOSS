from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

from hakken_ml_toolkit.tracker.impl.file_system import FSTracker, FSTrackerConfig


@pytest.fixture
def fs_tracker() -> FSTracker:
    config = FSTrackerConfig()
    return FSTracker(config)


@pytest.fixture
def temp_folder(tmp_path: Path) -> Path:
    return tmp_path


def test_track_value(
    fs_tracker: FSTracker, monkeypatch: pytest.MonkeyPatch, temp_folder: Path
) -> None:
    monkeypatch.setattr(fs_tracker, "folder", lambda: temp_folder)
    monkeypatch.setattr(fs_tracker, "persistance_is_enabled", True)

    fs_tracker.track_value("test_key", 42, step=1)

    file_path = temp_folder / "test_key.txt"
    assert file_path.exists()
    with open(file_path) as f:
        content = f.read()
    assert '{"test_key": 42, "step": 1}' in content


def test_track_config(
    fs_tracker: FSTracker, monkeypatch: pytest.MonkeyPatch, temp_folder: Path
) -> None:
    monkeypatch.setattr(fs_tracker, "folder", lambda: temp_folder)
    monkeypatch.setattr(fs_tracker, "persistance_is_enabled", True)

    config = {"param1": "value1", "param2": "value2"}
    fs_tracker.track_config(config)

    file_path: Path = temp_folder / "config.yaml"
    assert file_path.exists()
    with open(file_path) as f:
        loaded_config = yaml.safe_load(f)
    assert loaded_config == config


def test_track_data(
    fs_tracker: FSTracker, monkeypatch: pytest.MonkeyPatch, temp_folder: Path
) -> None:
    monkeypatch.setattr(fs_tracker, "folder", lambda: temp_folder)
    monkeypatch.setattr(fs_tracker, "persistance_is_enabled", True)

    data = {"key1": 10, "key2": torch.tensor(20.0)}
    fs_tracker.track_data(data, step=2)

    for key in data:
        file_path = temp_folder / f"{key}.txt"
        assert file_path.exists()
        with open(file_path) as f:
            content = f.read()
        assert f'"{key}": ' in content
        assert '"step": 2' in content


def test_track_table(
    fs_tracker: FSTracker, monkeypatch: pytest.MonkeyPatch, temp_folder: Path
) -> None:
    monkeypatch.setattr(fs_tracker, "folder", lambda: temp_folder)
    monkeypatch.setattr(fs_tracker, "persistance_is_enabled", True)

    columns = ["col1", "col2"]
    data = [[1, 2], [3, 4]]
    fs_tracker.track_table("test_table", columns, data, step=3)

    file_path = temp_folder / "test_table__3.tsv"
    assert file_path.exists()
    df = pd.read_csv(file_path, sep="\t")
    assert list(df.columns) == columns
    assert df.values.tolist() == data


def test_disabled_tracker(
    fs_tracker: FSTracker, monkeypatch: pytest.MonkeyPatch, temp_folder: Path
) -> None:
    monkeypatch.setattr(fs_tracker, "folder", lambda: temp_folder)
    monkeypatch.setattr(fs_tracker, "persistance_is_enabled", False)

    fs_tracker.track_value("test_key", 42)
    fs_tracker.track_config({"param": "value"})
    fs_tracker.track_data({"key": 10})
    fs_tracker.track_table("test_table", ["col"], [[1]])

    assert len(list(temp_folder.iterdir())) == 0


if __name__ == "__main__":
    pytest.main()
