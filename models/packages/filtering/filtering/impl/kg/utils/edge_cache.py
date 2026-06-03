from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from filtering.core.entities.kg import EdgeDirection, NodeId, Year, YearRange

if TYPE_CHECKING:
    import os


class EdgeCache:
    def __init__(self) -> None:
        self._edge_cache_dict: dict[EdgeDirection, dict[NodeId, set[Year]]] = {}

    def has_node_id(
        self,
        node_id: NodeId,
        direction: Literal[EdgeDirection.IN, EdgeDirection.OUT],
    ) -> bool:
        return direction in self._edge_cache_dict and node_id in self._edge_cache_dict[direction]

    def add_edge(
        self,
        node_id: NodeId,
        year: Year,
        direction: Literal[EdgeDirection.IN, EdgeDirection.OUT],
    ) -> None:
        return self.add_edges(node_id=node_id, years=[year], direction=direction)

    def add_edges(
        self,
        node_id: NodeId,
        years: list[Year],
        direction: Literal[EdgeDirection.IN, EdgeDirection.OUT],
    ) -> None:
        self._edge_cache_dict.setdefault(direction, {}).setdefault(node_id, set())
        self._edge_cache_dict[direction][node_id].update(years)

    def get_degree(
        self,
        node_id: NodeId,
        direction: Literal[EdgeDirection.IN, EdgeDirection.OUT],
        year_range: YearRange | None = None,
    ) -> int:
        if direction not in self._edge_cache_dict:
            return 0

        if node_id in self._edge_cache_dict[direction]:
            years = self._edge_cache_dict[direction][node_id]
            if year_range is None:
                return len(years)
            return len([year for year in years if year_range.start <= year < year_range.end])
        return 0

    @classmethod
    def from_pickle_directory(cls, directory: str | os.PathLike) -> EdgeCache:
        obj = cls()

        for path in Path(directory).glob("*.pkl"):
            filename = path.stem
            edge_direction_value, _ = filename.split("_", maxsplit=1)
            edge_direction = EdgeDirection(edge_direction_value)

            with open(path, "rb") as f:
                obj._edge_cache_dict[edge_direction] = pickle.load(f)

        if not obj._edge_cache_dict:
            raise ValueError("Edge cache is empty. Make sure that the path is correct.")

        return obj

    @classmethod
    def from_ndjson_directory(cls, directory: str | os.PathLike) -> EdgeCache:
        obj = cls()

        for path in Path(directory).glob("*.ndjson"):
            filename = path.stem
            edge_direction_value, _ = filename.split("_", maxsplit=1)
            edge_direction = EdgeDirection(edge_direction_value)

            obj._edge_cache_dict.setdefault(edge_direction, {})
            with open(path) as f:
                for line in f:
                    node_dict = json.loads(line)
                    node_id = node_dict["node_id"]
                    years = set(node_dict["years"])
                    obj._edge_cache_dict[edge_direction][node_id] = years

        if not obj._edge_cache_dict:
            raise ValueError("Edge cache is empty. Make sure that the path is correct.")

        return obj
