from abc import ABC, abstractmethod
from pathlib import Path
from typing import TypedDict, Unpack, cast

import networkx as nx
import torch
from hakken_ml_toolkit.ml_base_structures.fact import FactIndexList
from loguru import logger

from hakken_explainer.exceptions import MissingRequiredArgumentError, SetupNotCalledError
from hakken_explainer.utils import ExplainerUtils


class SetupKwargsBase(TypedDict, total=False):
    """Base kwargs for CandidateFinder setup."""

    facts_batch: torch.Tensor
    cache_folder: Path | None


class CandidateFinder(ABC):
    def __init__(
        self,
        max_candidates: int = 50000,
        undirected: bool = True,
    ) -> None:
        """k is the path length measured in edges. k=2 means two edges"""
        self._known_facts: torch.Tensor | None = None
        self._digraph: nx.MultiDiGraph
        self._graph: nx.MultiGraph | None = None
        self.device: str | torch.device = "cpu"

        self.max_candidates = max_candidates
        self.undirected = undirected

    @property
    def known_facts(self) -> torch.Tensor:
        if self._known_facts is None:
            raise SetupNotCalledError()
        return self._known_facts

    @property
    def known_graph(self) -> nx.MultiGraph:
        if self._graph is None:
            raise SetupNotCalledError()
        return self._graph

    @property
    def known_digraph(self) -> nx.MultiGraph:
        if self._digraph is None:
            raise SetupNotCalledError()
        return self._digraph

    def to_device(self, device: str | torch.device) -> None:
        self.device = device

    def get_known_digraph_relations(self, source: int, target: int) -> list[int]:
        edge_data = self.known_digraph.get_edge_data(source, target)
        if edge_data is None:
            return []

        return [data["relation"] for key, data in edge_data.items()]

    def known_digraph_has_edge(self, source: int, target: int, relation: int) -> bool:
        edge_data = self.known_digraph.get_edge_data(source, target)
        if edge_data is not None:
            for _key, data in edge_data.items():
                if data["relation"] == relation:
                    return True

        return False

    def get_known_edges(self) -> torch.Tensor:
        return cast("torch.Tensor", torch.unique(self.known_facts[:, [0, 2]], dim=0, sorted=False))

    def setup_known_facts(self, facts_batch: torch.Tensor) -> None:
        self._known_facts = facts_batch.to(self.device)

    def setup(self, **kwargs: Unpack[SetupKwargsBase]) -> None:
        """Build NetworkX graph from facts_batch."""

        facts_batch: torch.Tensor | None = kwargs.get("facts_batch")
        cache_folder: Path | None = kwargs.get("cache_folder")

        if facts_batch is None:
            raise MissingRequiredArgumentError(argument_name="facts_batch")

        self.setup_known_facts(facts_batch)

        file_path: Path | None = None
        file_path_u: Path | None = None
        if cache_folder:
            file_path = cache_folder / "graph.pkl"
            file_path_u = cache_folder / "undirected_graph.pkl"

        if file_path and file_path.exists():
            self._digraph = cast("nx.MultiDiGraph", ExplainerUtils.load_graph(file_path))

        else:
            logger.info(f"{file_path} not found")
            facts_list = facts_batch.tolist()
            logger.info(f"Constructing NetworkX directed graph from {len(facts_list)} edges")
            digraph: nx.MultiDiGraph = nx.MultiDiGraph()
            for subject, relation, obj in facts_list:
                digraph.add_edge(subject, obj, relation=relation)

            self._digraph = digraph
            if cache_folder and file_path is not None:
                cache_folder.mkdir(parents=True, exist_ok=True)
                ExplainerUtils.save_graph(self._digraph, file_path=file_path)

        if self.undirected:
            logger.info("Converting to undirected graph")
            if file_path_u is not None and file_path_u.exists():
                self._graph = cast("nx.MultiGraph", ExplainerUtils.load_graph(file_path_u))
            else:
                self._graph = self._digraph.to_undirected()
                if file_path_u is not None:
                    ExplainerUtils.save_graph(self._graph, file_path=file_path_u)

    @abstractmethod
    def find_candidates(
        self,
        source: int,
        target: int,
        relation: int | None = None,
        k: int | None = None,
        allowed_relations: list[int] | None = None,
    ) -> list[FactIndexList]:
        """a list of candidate explanations. Each candidate explanation has shape [num_facts, 3]"""
        pass
