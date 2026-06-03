from pathlib import Path


def repo_folder() -> Path:
    return Path(__file__).absolute().parent.parent.parent
