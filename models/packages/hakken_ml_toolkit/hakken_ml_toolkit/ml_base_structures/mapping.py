from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from hakken_ml_toolkit.ml_base_structures.common.exceptions import MappingNotFoundError


@dataclass
class Mapping:
    index_to_id: dict[int, str]
    id_to_index: dict[str, int]

    @classmethod
    def identity(cls, num_elements: int) -> Mapping:
        index_to_id = {i: str(i) for i in range(num_elements)}

        # Create the id_to_index dictionary
        id_to_index = {str(i): i for i in range(num_elements)}

        return cls(index_to_id=index_to_id, id_to_index=id_to_index)

    def encode(self, id_: str) -> int:
        if id_ not in self.id_to_index:
            msg = f"Unknown ID: {id_!r}"
            raise KeyError(msg)
        return self.id_to_index[id_]

    def decode(self, idx: int) -> str:
        if idx not in self.index_to_id:
            msg = f"Unknown index: {idx}"
            raise KeyError(msg)
        return self.index_to_id[idx]

    def __len__(self) -> int:
        return len(self.id_to_index)

    def get_ids(self, sort_by_key: bool = True) -> list[str]:
        if sort_by_key:
            return sorted(self.id_to_index.keys())
        return list(self.id_to_index.keys())

    def get_indexes(self, sort_by_key: bool = True) -> list[int]:
        if sort_by_key:
            return sorted(self.index_to_id.keys())
        return list(self.index_to_id.keys())

    @classmethod
    def load(cls, file_path_no_ext: Path) -> Mapping:
        id_to_index_path = file_path_no_ext.with_suffix(".tsv")
        if not id_to_index_path.exists():
            raise MappingNotFoundError(id_to_index_path)

        index_to_id = {}
        id_to_index = {}

        with open(id_to_index_path) as f:
            for line in f:
                key, value = line.strip().split("\t")
                id_to_index[key] = int(value)
                index_to_id[int(value)] = key

        return cls(index_to_id, id_to_index)

    def save(self, file_path_no_ext: Path) -> None:
        """
        Saves the id-to-index mapping as a tab-separated values (TSV) file.

        The file will be saved at: {file_path_no_ext}.tsv
        Each line contains an ID and its corresponding index, separated by a tab character.

        Args:
            folder (Path): Directory path where the TSV file will be saved.
                        If the directory doesn't exist, it will be created.

        Returns:
            None

        Example filepath:
            If folder = Path("/data/mappings/")
            File will be saved as: /data/mappings.tsv
        """

        id_to_index_path = file_path_no_ext.with_suffix(".tsv")
        id_to_index_path.parent.mkdir(parents=True, exist_ok=True)

        with open(id_to_index_path, "w") as f:
            for key, value in self.id_to_index.items():
                f.write(f"{key}\t{value}\n")
