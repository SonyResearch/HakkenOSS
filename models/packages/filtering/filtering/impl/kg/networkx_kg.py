from collections import defaultdict
from typing import Any

import networkx as nx
import pandas as pd
from loguru import logger

from filtering.core.contracts import KnowledgeGraph
from filtering.core.entities.config.knowledge_graph import NetworkXKnowledgeGraphConfig
from filtering.core.entities.kg import EdgeDirection, NodeId, YearRange


class NetworkXKnowledgeGraph(KnowledgeGraph[NetworkXKnowledgeGraphConfig]):
    def __init__(self, config: NetworkXKnowledgeGraphConfig):
        super().__init__(config)

        self._g: nx.MultiDiGraph = nx.MultiDiGraph()

        self._load()

    def _load(self) -> None:
        logger.info(f"Building a graph from {self.config.edges_path!s}")
        edges_dtype: defaultdict[str, type] = defaultdict(lambda: str)

        edges = pd.read_csv(str(self.config.edges_path), dtype=edges_dtype, delimiter="\t")
        edges["year"] = edges[self.config.edge_year_occurrences_column_name].str.split("|")
        edges = edges.explode("year")
        edges = edges.drop(self.config.edge_year_occurrences_column_name, axis=1).reset_index(
            drop=True
        )
        edges["year"] = edges["year"].astype(int)

        self._g = nx.from_pandas_edgelist(
            edges,
            self.config.edge_subject_id_column_name,
            self.config.edge_object_id_column_name,
            edge_attr=True,
            create_using=nx.MultiDiGraph,
        )

        logger.info(f"Adding node attributes from {self.config.nodes_path!s}")
        nodes_dtype: defaultdict[str, type] = defaultdict(lambda: str)

        nodes = pd.read_csv(str(self.config.nodes_path), dtype=nodes_dtype, delimiter="\t")
        for _, row in nodes.iterrows():
            node_index_column_name = self.config.node_id_column_name
            if node_index_column_name is None:
                raise ValueError("node_id_column_name is not configured")
            node_attrs: dict[str, Any] = {str(k): v for k, v in row.to_dict().items()}
            node_id = node_attrs[node_index_column_name]
            if self._g.has_node(node_id):
                self._g.nodes[node_id].update(node_attrs)
            else:
                self._g.add_node(node_id, **node_attrs)

    def get_degrees(
        self,
        node_ids: list[NodeId],
        direction: EdgeDirection,
        year_range: YearRange | None = None,
    ) -> list[int]:
        degrees: list[int] = []

        if year_range is None:
            g = self._g
        else:
            g = nx.subgraph_view(
                self._g,
                filter_edge=lambda s, o, k: (
                    year_range[0] <= self._g[s][o][k]["year"] < year_range[1]  # type: ignore
                ),
            )

        def degree_fn_factory():
            if direction == EdgeDirection.IN:
                return g.in_degree
            if direction == EdgeDirection.OUT:
                return g.out_degree
            return g.degree

        degree_fn = degree_fn_factory()

        for node_id in node_ids:
            if node_id not in self._g.nodes:
                raise KeyError(f"Node {node_id} does not exist in the graph.")
            degree = degree_fn(node_id)
            degrees.append(degree)
        return degrees
