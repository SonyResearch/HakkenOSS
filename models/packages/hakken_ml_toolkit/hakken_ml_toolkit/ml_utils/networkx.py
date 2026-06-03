import itertools
import random
from itertools import pairwise
from typing import Any, Protocol, cast

import networkx as nx
import numpy as np
import pandas as pd
from pydantic import BaseModel


def get_triples_between_nodes(graph: nx.Graph, source: str, target: str):
    triples = []
    if graph.has_edge(source, target):
        edge_info = graph.get_edge_data(source, target)
        for _, relation in edge_info.items():
            triples.append((source, relation, target))

    return list(triples) if triples else None


def compute_all_combinations_with_pandas(nested_list: list[list[Any]]):
    # Create a DataFrame with a single column containing all elements
    all_elements = [item for sublist in nested_list for item in sublist]
    df = pd.DataFrame({"elements": all_elements})

    # Create a column indicating which sublist each element came from
    sublist_indices = []
    for i, sublist in enumerate(nested_list):
        sublist_indices.extend([i] * len(sublist))
    df["sublist"] = sublist_indices

    # Generate all combinations by taking one element from each sublist
    groups = df.groupby("sublist")["elements"].apply(list).to_dict()

    # Use itertools.product to generate all combinations
    combinations = list(itertools.product(*[groups[i] for i in range(len(nested_list))]))

    # Convert to the desired output format
    return [list(combo) for combo in combinations]


def convert_path_to_triples(path: list[Any], graph: nx.Graph):
    all_triples = []  # List of all possible triples for each pair of nodes in the path

    # Process consecutive pairs of nodes
    for src, tgt in pairwise(path):
        # For each pair collect all possible triples
        triples = get_triples_between_nodes(graph, source=src, target=tgt)
        all_triples.append(triples)

    return compute_all_combinations_with_pandas(all_triples)


def triples_to_networkx(
    graph: nx.Graph,
    df_triples: pd.DataFrame,
    source_column: str,
    target_column: str,
    relation_column: str,
) -> nx.Graph:
    # Convert DataFrame directly to edge list using zip
    edges = zip(
        df_triples[source_column],
        df_triples[target_column],
        [{"relation": r} for r in df_triples[relation_column]],
        strict=False,
    )

    # Add edges in bulk
    graph.add_edges_from(edges)

    return graph


class NetworkXUtilsConfig(BaseModel):
    source_column: str = "source"
    target_column: str = "target"
    relation_column: str = "relation"
    multiple_edges: bool = True
    directed: bool = True


class NetworkXUtils(Protocol):
    @staticmethod
    def load_graph_from_pandas(df: pd.DataFrame, config: NetworkXUtilsConfig) -> nx.Graph:
        """Loads into a netwrokx graph a pandas dataframe

        Args:
            df: A pandas dataframe
            config: a pydantic config defining which columns to load and which graph to
                build
                - source_column: Name of the column with the source entity to place in the graph
                - target_colum: Name of the column with the target entity to place in the graph
                - relation_column: Name of the column with the relation type to place on the graph,
                    connecting from the source to the target
                - multiple_edges: wheather you want a graph with multiple edges between nodes
                - directed: wheater to take into consideration directionality

        Returns:
            nx.Graph: the chosen networkx graph type
        """

        source_column = config.source_column
        target_column = config.target_column
        relation_column = config.relation_column

        df_triples = df[[source_column, target_column, relation_column]]

        # Casting everything to a string to avoid confusion
        df_triples.loc[:, source_column] = df_triples[source_column].astype(str)
        df_triples.loc[:, target_column] = df_triples[target_column].astype(str)
        df_triples.loc[:, relation_column] = df_triples[relation_column].astype(str)
        graph: nx.MultiGraph | nx.Graph
        if config.multiple_edges:
            graph = nx.MultiDiGraph() if config.directed else nx.MultiGraph()
        else:
            graph = nx.DiGraph() if config.directed else nx.Graph()

        return triples_to_networkx(
            graph=graph,
            df_triples=df_triples,
            source_column=source_column,
            target_column=target_column,
            relation_column=relation_column,
        )

    @staticmethod
    def convert_graph_to_undirected(graph: nx.DiGraph, multiple_edges: bool = True) -> nx.Graph:
        graph_u: nx.Graph = nx.MultiGraph() if multiple_edges else nx.Graph()

        # Add edges from the original MultiDiGraph (or MultiGraph)
        for u, v, data in graph.edges(data=True):
            graph_u.add_edge(u, v, **data)

        return graph_u

    @staticmethod
    def get_shortest_path_length(  # noqa: PLR0913
        graph: nx.Graph,
        source: int | str,
        target: int | str,
        weight: str | None = None,
        method: str = "dijkstra",
        include_extrema=False,
    ) -> int:
        """Returns the length of the shortest path connecting a source and a target.
        Order of source and target does not matter for undirected graphs.

        Args:
            graph: a networkx directed or undirected graph
            source: Name of source entity to search
            target: Name of the target entity to search
            weight: ...
            method: ...
            include_extrema: if set to True, instead of counting the total number of links,
                also the source and target are counted. Defaults to False.

        Returns:
            int: the lenght of the path, If not found returns -1
        """

        path = NetworkXUtils.shortest_path(graph, source, target, weight, method)

        if len(path) == 0:
            return -1
        return len(path) - 1 if include_extrema is False else len(path) + 1

    @staticmethod
    def get_number_of_common_neighbors(
        graph: nx.Graph,
        source: int | str,
        target: int | str,
    ) -> int:
        neighbors = NetworkXUtils.common_neighbors(graph, source, target)

        return len(neighbors)

    @staticmethod
    def common_neighbors(graph: nx.Graph, source: int | str, target: int | str) -> list[int | str]:
        neighbors_source = set(graph.neighbors(source))
        neighbors_target = set(graph.neighbors(target))

        return list(neighbors_source.intersection(neighbors_target))

    @staticmethod
    def shortest_path(
        graph: nx.Graph,
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
    def all_shortest_paths(
        graph: nx.Graph,
        source: int | str,
        target: int | str,
        weight: str | None = None,
        method: str = "dijkstra",
    ) -> list[list[tuple[Any]]]:
        try:
            paths_list = nx.all_shortest_paths(
                graph, source=source, target=target, weight=weight, method=method
            )
            all_possible_paths = []

            for path in paths_list:
                all_paths = convert_path_to_triples(path, graph)
                all_possible_paths.extend(all_paths)

        except nx.NetworkXNoPath:
            return []
        else:
            assert isinstance(all_possible_paths, list)
            return all_possible_paths

    @staticmethod
    def get_personalized_pagerank(  # noqa: PLR0913
        graph: nx.Graph,
        source: int | str,
        target: int | str,
        n_simulations: int = 1000,
        max_steps: int | None = None,
        alpha: float = 0.85,  # damping factor
    ) -> float:
        """
        Calculate personalized PageRank / random walk proximity between two nodes.

        Args:
            graph: NetworkX MultiDiGraph
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
        graph: nx.Graph,
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
        graph: nx.Graph,
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
        graph: nx.DiGraph,
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
        graph: nx.DiGraph,
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
