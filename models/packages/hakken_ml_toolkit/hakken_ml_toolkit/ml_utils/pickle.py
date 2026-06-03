from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


class PickleUtils:
    """Utility class for loading and saving data using pickle."""

    @staticmethod
    def load(file_path: str | Path) -> Any:
        path = Path(file_path) if isinstance(file_path, str) else file_path
        if not path.exists():
            raise FileNotFoundError(path)

        with path.open("rb") as file:
            return pickle.load(file)

    @staticmethod
    def save(data: Any, file_path: str | Path) -> None:
        # Convert string path to Path object if necessary
        path = Path(file_path) if isinstance(file_path, str) else file_path

        # Create directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save the data to pickle file
        with path.open("wb") as file:
            pickle.dump(data, file)
