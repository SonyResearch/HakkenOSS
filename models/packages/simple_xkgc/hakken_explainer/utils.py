import pickle
from collections import deque
from pathlib import Path
from typing import Any, cast

import networkx as nx
from hakken_ml_toolkit.ml_base_structures import KnowledgeGraph
from loguru import logger


class ExplainerUtils:
    @staticmethod
    def load_graph(file_path: Path, format: str = "pickle") -> nx.Graph:
        logger.info(f"Loading graph from {file_path}")
        if format != "pickle":
            raise NotImplementedError()
        with open(file_path, "rb") as f:
            return cast("nx.Graph", pickle.load(f))

    @staticmethod
    def save_graph(graph: nx.Graph, file_path: Path, format: str = "pickle") -> None:
        logger.info(f"Saving graph to {file_path}")
        if format != "pickle":
            raise NotImplementedError()
        with open(file_path, "wb") as f:
            pickle.dump(graph, f)

    @staticmethod
    def find_k_length_paths(
        graph: nx.Graph, start: int, end: int, k: int, path: list[int] | None = None
    ) -> list[list[int]]:
        """
        Find all simple paths of exactly length k (number of edges) between start and end nodes
        in a graph.

        Args:
            graph (nx.Graph): The input graph (undirected or directed).
            start (int): The starting node for paths.
            end (int): The target ending node for paths.
            k (int): The exact length (number of edges) the paths should have.
            path (Optional[List[int]]): Internal parameter used for recursive path construction
            (default is None).

        Returns:
            List[List[int]]: A list of paths (each path is a list of nodes) where each path
                            has length k and starts at `start` and ends at `end`.

        Note:
            The paths returned are simple paths (no repeated nodes).
            Length is measured as the count of edges between nodes in the path.
        """
        if path is None:
            path = [start]
        if len(path) - 1 == k:
            if path[-1] == end:
                return [path]
            return []
        paths: list[list[int]] = []
        for neighbor in graph.neighbors(path[-1]):
            if neighbor not in path:  # avoid cycles
                newpaths = ExplainerUtils.find_k_length_paths(
                    graph, start, end, k, [*path, neighbor]
                )
                for p in newpaths:
                    paths.append(p)
        return paths

    @staticmethod
    def convert_triple_path_id_to_str(triple_path: list[tuple[Any, Any, Any]]) -> str:
        """
        Converts a path of triples into a human-readable string representation.

        Creates a string where each triple is formatted as "[subject-relation->object]"
        and triples are connected by " <> " to form a path.

        Args:
            triple_path: A list of (subject, relation, object) tuples representing
                        a path through a knowledge graph.

        Returns:
            A string representation of the path in a human-readable format.
        """
        readable_segments = []

        for subject, relation, object_entity in triple_path:
            readable_segments.append(f"[{subject}-{relation}->{object_entity}]")

        # Join the readable segments with the connector
        return " <> ".join(readable_segments)

    @staticmethod
    def convert_triple_path_id_to_index(
        triple_path: list[tuple[Any, Any, Any]], knowledge_graph: KnowledgeGraph
    ) -> list[tuple[int, int, int]]:
        """
        Converts a path of triples into numerical indices based on knowledge graph mappings.

        Maps each entity and relation in the path to its corresponding numerical index
        in the knowledge graph mappings.

        Args:
            triple_path: A list of (subject, relation, object) tuples representing
                    a path through the knowledge graph.
            knowledge_graph: The KnowledgeGraph object containing entity and relation mappings.

        Returns:
            A list of (subject_idx, relation_idx, object_idx) tuples with numerical indices.
        """
        indexed_triples = []

        for subject, relation, object_entity in triple_path:
            # Convert entities and relations to their numerical indices
            subject_idx = knowledge_graph.entity_mapping.id_to_index[subject]
            relation_idx = knowledge_graph.relation_mapping.id_to_index[relation]
            object_idx = knowledge_graph.entity_mapping.id_to_index[object_entity]

            # Store the indexed representation
            indexed_triples.append((subject_idx, relation_idx, object_idx))

        return indexed_triples

    @staticmethod
    def find_paths_length_k_bfs(graph: nx.Graph, source: Any, target: Any, k: int):
        """
        Custom BFS to find paths of exactly length k between two nodes
        More efficient for sparse graphs or when NetworkX has overhead
        """
        if k == 0:
            return [[source]] if source == target else []

        if k == 1:
            return [[source, target]] if graph.has_edge(source, target) else []

        # BFS with path tracking
        queue = deque([(source, [source], 0)])  # (current_node, path, length)
        valid_paths = []

        while queue:
            current_node, path, length = queue.popleft()

            # If we've reached the target with exactly k edges
            if current_node == target and length == k:
                valid_paths.append(path)
                continue

            # If path is already k edges long but not at target, skip
            if length >= k:
                continue

            # Explore neighbors
            for neighbor in graph.neighbors(current_node):
                if neighbor not in path:  # Avoid cycles (simple paths only)
                    new_path = [*path, neighbor]
                    queue.append((neighbor, new_path, length + 1))

        return valid_paths
