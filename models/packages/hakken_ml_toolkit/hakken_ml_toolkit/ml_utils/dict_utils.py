from __future__ import annotations

from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from pathlib import Path


class DictUtils:
    @staticmethod
    def flatten(d: dict, parent_key="", delimiter=".") -> dict[str, Any]:
        flattened = {}
        for key, value in d.items():
            new_key = f"{parent_key}{delimiter}{key}" if parent_key else key

            if isinstance(value, dict):
                flattened.update(DictUtils.flatten(value, new_key, delimiter))
            else:
                flattened[new_key] = value

        return flattened

    @staticmethod
    def to_txt(my_dict: dict[str, Any], file_path: str) -> str:
        with open(file_path, "w") as f:
            for key, value in my_dict.items():
                f.write(f"{key}\t{value}\n")
        return file_path

    @staticmethod
    def from_txt(file_path: Path) -> dict[str, Any]:
        result = {}
        with open(file_path) as f:
            for line in f:
                key, value = line.strip().split("\t", 1)
                result[key] = value
        return result

    @staticmethod
    def to_yaml(my_dict: dict[str, Any], file_path: str | Path) -> str:
        with open(file_path, "w") as f:
            yaml.dump(my_dict, f)
        return str(file_path)
