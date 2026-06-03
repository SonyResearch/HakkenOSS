import itertools

import networkx as nx
from hakken_ml_toolkit.ml_base_structures.fact import FactIndexList
from loguru import logger

from hakken_explainer.candidate_finder.base import CandidateFinder

MIN_NUMBER_OF_NODES = 2


class CorpusPathFinder(CandidateFinder):
    def find_candidates(
        self,
        source: int,
        target: int,
        relation: int | None = None,
        k: int | None = None,
        allowed_relations: list[int] | None = None,
    ) -> list[FactIndexList]:
        if relation is not None:
            logger.warning(f"Relation {relation} is not used for finding the candidates")

        node_paths = self.find_node_paths(source=source, target=target, k=k)
        if len(node_paths) == 0:
            return []

        result = []
        for node_path in node_paths:
            fact_paths = self._get_fact_paths(node_path, allowed_relations)
            result.extend(fact_paths)

        logger.debug(f"Found {len(result)} FactIndexPathway")
        return result

    def find_node_paths(self, source: int, target: int, k: int | None = None) -> list:
        logger.debug(f"Finding paths from {source} to {target} of length {k}")

        shortest_path_len = self.shortest_path_length(source, target)
        if shortest_path_len == -1:
            return []

        logger.debug(f"Shortest path length is {shortest_path_len}")

        if k is None:
            k = shortest_path_len

        if k == -1 or source not in self.known_graph or target not in self.known_graph:
            return []

        try:
            if k == shortest_path_len:
                node_paths = list(
                    nx.all_shortest_paths(self.known_graph, source=source, target=target)
                )
            else:
                logger.debug("Using all simple paths")
                num_nodes_per_path = k + 1
                node_paths = []
                paths_iter = nx.all_simple_paths(
                    self.known_graph, source=source, target=target, cutoff=k
                )
                for path in itertools.islice(paths_iter, self.max_candidates):
                    if len(path) == num_nodes_per_path:
                        node_paths.append(path)

        except nx.NetworkXNoPath:
            logger.warning("No paths found")
            return []

        logger.debug(f"Found {len(node_paths)} node paths")
        return node_paths

    def _get_fact_paths(
        self, node_path: list[int], allowed_relations: list[int] | None = None
    ) -> list[FactIndexList]:
        """Get all possible relation sequences for a given node path."""
        if len(node_path) < MIN_NUMBER_OF_NODES:
            return []

        relations_per_hop = []

        for i in range(len(node_path) - 1):
            u, v = node_path[i], node_path[i + 1]
            relations_uv = self.get_known_digraph_relations(u, v)
            relations_vu = self.get_known_digraph_relations(v, u)

            hop_relations = list(set(relations_uv) | set(relations_vu))

            if len(hop_relations) == 0:
                msg = f"Edge data for ({u}, {v}) and ({v}, {u}) is None"
                raise RuntimeError(msg)

            if allowed_relations is not None:
                hop_relations = [r for r in hop_relations if r in allowed_relations]

            if not hop_relations:
                return []

            relations_per_hop.append(hop_relations)

        relation_sequences = list(itertools.product(*relations_per_hop))

        fact_paths = []
        for relation_seq in relation_sequences:
            facts_per_hop = []

            relation_seq_is_valid = True

            for i in range(len(relation_seq)):
                u = node_path[i]
                v = node_path[i + 1]
                r = relation_seq[i]

                hop_facts = []
                if self.known_digraph_has_edge(u, v, r):
                    hop_facts.append((u, r, v))
                if self.known_digraph_has_edge(v, u, r):
                    hop_facts.append((v, r, u))

                if not hop_facts:
                    relation_seq_is_valid = False
                    break

                facts_per_hop.append(hop_facts)

            if not relation_seq_is_valid:
                continue
            # Generate all combinations of fact directions
            for fact_combination in itertools.product(*facts_per_hop):
                fact_paths.append(list(fact_combination))

        return fact_paths

    def shortest_path_length(self, source: int, target: int) -> int:
        """
        Returns the length of the shortest path from source to target.
        Returns -1 if no path exists.
        """
        # Handle edge cases
        if source not in self.known_graph or target not in self.known_graph:
            return -1

        if source == target:
            return 0

        try:
            return int(nx.shortest_path_length(self.known_graph, source=source, target=target))
        except nx.NetworkXNoPath:
            return -1
