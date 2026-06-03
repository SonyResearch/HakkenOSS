import itertools
import random
from itertools import pairwise
from typing import Any, Protocol, cast

import networkx as nx
import numpy as np


class NetworkXUtils(Protocol):
    @staticmethod
    def extract_path_triple_combinations(
        graph: nx.Graph, path: list[Any], inverse: bool = False
    ) -> list[list[tuple]]:
        """Extract all possible triple combinations along a path in a graph.

        For each adjacent node pair in the path, finds all possible
        (subject, relation, object) triples from the graph edges.
        Then computes all combinations of these triples to form complete paths.

        Args:
            graph (nx.Graph): A NetworkX graph (MultiDiGraph, DiGraph, MultiGraph, or Graph).
            path (list[Any]): A sequence of nodes representing a path in the graph.
            inverse (bool, optional): If True, also consider edges in the reverse direction
                between adjacent nodes. Defaults to False.

        Returns:
            list: A list of triple paths, where each triple path is a list of
                (subject, relation, object) tuples representing one possible way
                to traverse the path using available edges.

        Examples:
            >>> import networkx as nx
            >>> G = nx.MultiDiGraph()
            >>> # Add edges with different relation types
            >>> G.add_edge('A', 'B', relation='connects_to')
            >>> G.add_edge('A', 'B', relation='leads_to')
            >>> G.add_edge('B', 'C', relation='part_of')
            >>> G.add_edge('C', 'B', relation='contains')
            >>> # Define a path through the graph
            >>> path = ['A', 'B', 'C']
            >>> # Get all possible triple combinations for this path
            >>> result = extract_path_triple_combinations(G, path)
            >>> print(result)
            [[('A', 'connects_to', 'B'), ('B', 'part_of', 'C')],
            [('A', 'leads_to', 'B'), ('B', 'part_of', 'C')]]
            >>> # With inverse=True, also consider reverse edges
            >>> result_with_inverse = extract_path_triple_combinations(G, path, inverse=True)
            >>> print(result_with_inverse)
            [[('A', 'connects_to', 'B'), ('B', 'part_of', 'C')],
            [('A', 'leads_to', 'B'), ('B', 'part_of', 'C')],
            [('A', 'connects_to', 'B'), ('C', 'contains', 'B')],
            [('A', 'leads_to', 'B'), ('C', 'contains', 'B')]]
        """
        if len(path) < 2:
            return []

        # List of edge triple options between each adjacent node pair
        edge_triple_options = []

        for source_node, target_node in pairwise(path):
            node_pair_triples = []

            # Check for outgoing edges (source -> target)
            if graph.has_edge(source_node, target_node):
                edge_relations = list(graph.get_edge_data(source_node, target_node).keys())
                for relation in edge_relations:
                    node_pair_triples.append((source_node, relation, target_node))

            # Check for incoming edges (target -> source)
            if inverse and graph.has_edge(target_node, source_node):
                edge_relations = list(graph.get_edge_data(target_node, source_node).keys())
                for relation in edge_relations:
                    node_pair_triples.append((target_node, relation, source_node))

            # Ensure there's at least one triple connecting these nodes
            if not node_pair_triples:
                msg = f"""
                No connecting edges found between nodes '{source_node}' and '{target_node}'.
                Adjacent nodes in the path must have at least one connecting edge in 
                either direction."""
                raise ValueError(msg)

            edge_triple_options.append(node_pair_triples)

        # Generate all possible combinations of triples (one from each node pair)
        return list(itertools.product(*edge_triple_options))  # type: ignore

    @staticmethod
    def get_triples_between_nodes(
        graph: nx.Graph, subject: str, object: str, inverse: bool = False
    ) -> list[tuple[Any, Any, Any]]:
        """Retrieve all triples (subject-relation-object) between two specified nodes in a graph.

        Args:
            graph (nx.Graph): The graph to search in.
            subject (str): The source node.
            object (str): The target node.
            inverse (bool, optional): If True, also search for triples in the reverse direction
                (from object to subject). Defaults to False.

        Returns:
            list[tuple[Any, Any, Any]]: A list of triples, where each triple is
                (node1, relation, node2).

        Examples:
            >>> import networkx as nx
            >>> G = nx.DiGraph()
            >>> G.add_edge('person', 'city', relation='lives_in')
            >>> G.add_edge('city', 'person', relation='has_resident')
            >>> get_triples_between_nodes(G, 'person', 'city')
            [('person', 'lives_in', 'city')]
            >>> get_triples_between_nodes(G, 'person', 'city', inverse=True)
            [('person', 'lives_in', 'city'), ('city', 'has_resident', 'person')]
        """
        triples = []
        if graph.has_edge(subject, object):
            edge_info = graph.get_edge_data(subject, object)
            for _, relation in edge_info.items():
                triples.append((subject, relation, object))
        if inverse and graph.has_edge(object, subject):
            edge_info = graph.get_edge_data(object, subject)
            for _, relation in edge_info.items():
                triples.append((object, relation, subject))
        return triples

    @staticmethod
    def get_shortest_path_length(
        graph: nx.DiGraph,
        source: int | str,
        target: int | str,
        weight: str | None = None,
        method: str = "dijkstra",
    ) -> int:
        path = NetworkXUtils.shortest_path(graph, source, target, weight, method)
        if len(path) == 0:
            return -1
        return len(path)

    @staticmethod
    def get_number_of_common_neighbors(
        graph: nx.DiGraph,
        source: int | str,
        target: int | str,
    ) -> int:
        neighbors = NetworkXUtils.common_neighbors(graph, source, target)

        return len(neighbors)

    @staticmethod
    def common_neighbors(
        graph: nx.DiGraph, source: int | str, target: int | str
    ) -> list[int | str]:
        neighbors_source = set(graph.neighbors(source))
        neighbors_target = set(graph.neighbors(target))

        return list(neighbors_source.intersection(neighbors_target))

    @staticmethod
    def shortest_path(
        graph: nx.DiGraph,
        source: int | str,
        target: int | str,
        weight: str | None = None,
        method: str = "dijkstra",
    ) -> list[int | str]:
        try:
            shortest_paths = nx.shortest_path(
                graph, source=source, target=target, weight=weight, method=method
            )

        except nx.NetworkXNoPath:
            return []
        else:
            assert isinstance(shortest_paths, list)
            return shortest_paths

    @staticmethod
    def get_personalized_pagerank(  # noqa: PLR0913
        graph: nx.DiGraph,
        source: int | str,
        target: int | str,
        n_simulations: int = 1000,
        max_steps: int | None = None,
        alpha: float = 0.85,  # damping factor
    ) -> float:
        """
        Calculate personalized PageRank / random walk proximity between two nodes.

        Args:
            graph: NetworkX DiGraph
            source: Starting node for random walks
            target: Target node to measure proximity to
            n_simulations: Number of random walks to perform
            max_steps: Maximum steps per walk (defaults to 3 * number of nodes)
            alpha: Damping factor (probability of continuing the walk)

        Returns:
            float: Proximity score between 0 and 1
        """
        if max_steps is None:
            max_steps = 3 * len(graph)

        # Check if nodes exist in graph
        if source not in graph or target not in graph:
            return 0.0

        def single_random_walk() -> float:
            """Perform single random walk and track target node visits."""
            current = source
            target_visits = 0
            total_steps = 0

            while total_steps < max_steps:
                # Random jump with probability (1-alpha)
                if np.random.random() > alpha:
                    current = source  # Reset to source node
                    continue

                # Get neighbors of current node
                neighbors = list(graph.neighbors(current))
                if not neighbors:
                    break

                # Random walk step
                current = np.random.choice(neighbors)
                total_steps += 1

                # Count visits to target
                if current == target:
                    target_visits += 1

            return target_visits / max_steps if total_steps > 0 else 0

        # Perform multiple random walks and average results
        proximity_scores = [single_random_walk() for _ in range(n_simulations)]

        # Return average proximity score
        return float(np.mean(proximity_scores))

    @staticmethod
    def get_comute_time(
        graph: nx.DiGraph,
        source: int | str,
        target: int | str,
        n_simulations: int = 1000,
        max_steps: int | None = None,
    ) -> float | None:
        if not nx.has_path(graph, source, target) or not nx.has_path(graph, target, source):
            return None

        # Simulate forward path (source to target)
        forward_times = []
        for _ in range(n_simulations):
            steps = NetworkXUtils.simulate_random_walk(graph, source, target, max_steps)
            if max_steps:  # Only count successful walks
                forward_times.append(steps)

        # Simulate return path (target to source)
        return_times = []
        for _ in range(n_simulations):
            steps = NetworkXUtils.simulate_random_walk(graph, target, source)
            if max_steps is not None and steps < max_steps:  # Only count successful walks
                return_times.append(steps)

        # Check if we have enough successful simulations
        if not forward_times or not return_times:
            return None

        # Calculate average commute time
        avg_forward = np.mean(forward_times)
        avg_return = np.mean(return_times)
        return float(avg_forward + avg_return)

    @staticmethod
    def simulate_random_walk(
        graph: nx.DiGraph,
        start: int | str,
        end: int | str,
        max_steps: int | None = None,
    ) -> int:
        """Simulate single random walk between start and end nodes."""
        current = start
        steps = 0
        if max_steps is None:
            max_steps = len(graph) * 100  # Prevent infinite loops

        while current != end and steps < max_steps:
            # Get neighbors of current node
            neighbors = list(graph.neighbors(current))
            if not neighbors:
                return max_steps

            # Randomly select next node
            current = np.random.choice(neighbors)
            steps += 1

        return steps if steps < max_steps else max_steps

    @staticmethod
    def get_ego_network(
        graph: nx.MultiDiGraph,
        node: int,
        radius: int = 1,
        node_proportion: float | None = None,
    ) -> nx.MultiDiGraph:
        nodes = {node}
        current_nodes = {node}

        for _ in range(radius):
            next_nodes: set[Any] = set()
            for n in current_nodes:
                next_nodes.update(graph.predecessors(n))
                next_nodes.update(graph.successors(n))
            current_nodes = next_nodes - nodes
            if node_proportion is not None and len(current_nodes) > 0:
                k = max(int(len(current_nodes) * node_proportion), 1)
                current_nodes = set(random.choices(list(current_nodes), k=k))

            nodes.update(current_nodes)
        # Return induced subgraph
        return cast("nx.MultiDiGraph", graph.subgraph(nodes).copy())

    @staticmethod
    def get_random_walks_subgraph(
        graph: nx.MultiDiGraph,
        start_node: int,
        num_walks: int = 5,
        walk_length: int = 3,
        seed: int | None = None,
    ) -> nx.MultiDiGraph:
        if seed is not None:
            random.seed(seed)

        sub_graph: nx.MultiDiGraph = nx.MultiDiGraph()
        sub_graph.add_node(start_node, **graph.nodes[start_node])
        edge_keys_added = set()
        for _ in range(num_walks):
            current = start_node

            for _ in range(walk_length):
                edges = []
                for s in graph.successors(current):
                    edges.append((current, s))
                for p in graph.predecessors(current):
                    edges.append((p, current))
                if not edges:
                    break

                edge = random.choice(edges)
                edge_data = graph.get_edge_data(edge[0], edge[1])

                available_keys = [
                    k for k in edge_data if (edge[0], edge[1], k) not in edge_keys_added
                ]
                current = edge[0] if edge[0] != current else edge[1]

                if available_keys:
                    edge_key = random.choice(available_keys)
                    edge_keys_added.add((edge[0], edge[1], edge_key))

                    sub_graph.add_edge(edge[0], edge[1], **edge_data[edge_key])

                    sub_graph.add_node(current, **graph.nodes[current])

        return sub_graph
