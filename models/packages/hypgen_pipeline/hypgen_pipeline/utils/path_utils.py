from pathlib import Path


def repo_folder() -> Path:
    return Path(__file__).absolute().parent.parent.parent


def data_folder() -> Path:
    return repo_folder() / "data"
