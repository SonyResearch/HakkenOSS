import textwrap
from collections import defaultdict
from typing import Literal

from loguru import logger
from neo4j import GraphDatabase

from filtering.core.contracts import KnowledgeGraph
from filtering.core.entities.config.knowledge_graph import Neo4jKnowledgeGraphConfig
from filtering.core.entities.kg import EdgeDirection, NodeId, YearRange
from filtering.impl.kg.utils.edge_cache import EdgeCache


class Neo4jKnowledgeGraph(KnowledgeGraph[Neo4jKnowledgeGraphConfig]):
    _QUERY_TEMPLATE: str = textwrap.dedent(
        """\
        MATCH (s)-[r]->(o)
        WHERE {condition}
        RETURN {return_clause}
        """,
    )

    def __init__(self, config: Neo4jKnowledgeGraphConfig) -> None:
        super().__init__(config)

        self.driver = GraphDatabase.driver(
            self.config.base_url, auth=(self.config.username, self.config.password)
        )

        self._edge_cache: EdgeCache = EdgeCache()

    @staticmethod
    def _compose_query(
        direction: EdgeDirection,
        node_ids: list[NodeId],
    ) -> str:
        field_of_interest = "o.node_id" if direction == EdgeDirection.IN else "s.node_id"

        condition_str = f"{field_of_interest} IN {node_ids}"

        return_values = [
            f"{field_of_interest} AS node_id",
            "r.year_occurrences AS year_occurrences",
        ]
        return_clause = ", ".join(return_values)

        return Neo4jKnowledgeGraph._QUERY_TEMPLATE.format(
            condition=condition_str, return_clause=return_clause
        )

    def _execute_query_and_add_to_cache(
        self, query: str, direction: Literal[EdgeDirection.IN, EdgeDirection.OUT]
    ) -> None:
        with self.driver.session() as session:
            result = session.run(query)

        for row in result:
            node_id = row["node_id"]
            year_occurrences = [int(year) for year in row["year_occurrences"].split("|")]

            self._edge_cache.add_edges(node_id=node_id, years=year_occurrences, direction=direction)

    def get_degrees(
        self,
        node_ids: list[NodeId],
        direction: EdgeDirection,
        year_range: YearRange | None = None,
    ) -> list[int]:
        node_ids_to_query_in = [
            ocid
            for ocid in node_ids
            if not self._edge_cache.has_node_id(node_id=ocid, direction=EdgeDirection.IN)
        ]
        node_ids_to_query_out = [
            ocid
            for ocid in node_ids
            if not self._edge_cache.has_node_id(node_id=ocid, direction=EdgeDirection.OUT)
        ]

        in_query = Neo4jKnowledgeGraph._compose_query(
            direction=EdgeDirection.IN, node_ids=node_ids_to_query_in
        )
        out_query = Neo4jKnowledgeGraph._compose_query(
            direction=EdgeDirection.OUT, node_ids=node_ids_to_query_out
        )

        if direction in (EdgeDirection.IN, EdgeDirection.ALL) and node_ids_to_query_in:
            logger.info(f"Querying to Octopus with: {in_query}")
            self._execute_query_and_add_to_cache(query=in_query, direction=EdgeDirection.IN)

        if direction in (EdgeDirection.OUT, EdgeDirection.ALL) and node_ids_to_query_out:
            logger.info(f"Querying to Octopus with: {out_query}")
            self._execute_query_and_add_to_cache(query=out_query, direction=EdgeDirection.OUT)

        degree_dict: dict[NodeId, int] = defaultdict(int)

        if direction in (EdgeDirection.IN, EdgeDirection.ALL):
            for node_id in node_ids:
                degree_dict[node_id] += self._edge_cache.get_degree(
                    node_id=node_id, direction=EdgeDirection.IN, year_range=year_range
                )
        if direction in (EdgeDirection.OUT, EdgeDirection.ALL):
            for node_id in node_ids:
                degree_dict[node_id] += self._edge_cache.get_degree(
                    node_id=node_id, direction=EdgeDirection.OUT, year_range=year_range
                )

        return [degree_dict[ocid] for ocid in node_ids]
