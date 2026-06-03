import itertools

import networkx as nx
from loguru import logger

NodeID = str
RelationID = str


class FactPathFinder:
    def __init__(self, graph: nx.MultiDiGraph, relation_type_attr: str = "relation_type") -> None:
        self.graph = graph
        self.relation_type_attr = relation_type_attr

    def get_known_relations(self, source: NodeID, target: NodeID) -> set[RelationID]:
        if not self.graph.has_edge(source, target):
            return set()
        return set(self.graph[source][target].keys())

    def has_edge(self, source: NodeID, target: NodeID, relation: RelationID) -> bool:
        return self.graph.has_edge(source, target, key=relation)

    def node_path_to_facts(  # noqa: PLR0912
        self,
        node_path: list[NodeID],
        allowed_relations: list[RelationID] | None = None,
        max_paths: int | None = None,
        include_reverse_each_hop: bool = True,
    ) -> list[list[tuple[NodeID, RelationID, NodeID]]]:
        if len(node_path) < 2:
            logger.warning(f"Node path too short (len={len(node_path)}), returning empty result")
            return []

        if max_paths is not None and max_paths == 0:
            return []

        relations_per_hop = []

        for i in range(len(node_path) - 1):
            u, v = node_path[i], node_path[i + 1]
            relations_uv = self.get_known_relations(u, v)
            hop_relations: list
            if include_reverse_each_hop:
                relations_vu = self.get_known_relations(v, u)
                hop_relations = list(relations_uv | relations_vu)
            else:
                hop_relations = list(relations_uv)

            if len(hop_relations) == 0:
                msg = f"Edge data for ({u}, {v}) and ({v}, {u}) is None"
                raise RuntimeError(msg)

            if allowed_relations is not None:
                hop_relations = [r for r in hop_relations if r in allowed_relations]

            if not hop_relations:
                return []

            relations_per_hop.append(hop_relations)

        fact_paths = []
        for relation_seq in itertools.product(*relations_per_hop):
            facts_per_hop = []

            relation_seq_is_valid = True

            for i in range(len(relation_seq)):
                u = node_path[i]
                v = node_path[i + 1]
                r = relation_seq[i]

                hop_facts = []
                if self.has_edge(u, v, r):
                    hop_facts.append((u, r, v))
                if include_reverse_each_hop and self.has_edge(v, u, r):
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
                if max_paths is not None and len(fact_paths) >= max_paths:
                    logger.info(f"Reached max_paths limit ({max_paths}), returning early")
                    return fact_paths
        logger.info(f"Generated {len(fact_paths)} fact paths from node path")
        return fact_paths
