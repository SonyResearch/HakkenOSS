from __future__ import annotations

import time
from typing import TYPE_CHECKING

from loguru import logger
from neo4j import GraphDatabase
from query_common.entities.kg.concept import Concept
from query_common.entities.kg.triple import Triple

from complex_query.core.contracts.kg import KnowledgeGraph
from complex_query.core.entities.config.kg import Neo4jKGConfig

if TYPE_CHECKING:
    from query_common.entities.kg.identifier import (
        ConceptIdentifier,
        DomainIdentifier,
        RelationIdentifier,
    )


class Neo4jKG(KnowledgeGraph[Neo4jKGConfig]):
    def __init__(self, config: Neo4jKGConfig) -> None:
        super().__init__(config)

        self.driver = GraphDatabase.driver(
            self.config.base_url, auth=(self.config.username, self.config.password)
        )

    def _execute_query(self, query: str) -> list[dict]:
        with self.driver.session() as session:
            query_result = session.run(query)

            results = query_result.data()
            summary = query_result.consume()
            logger.info(f"Query took {summary.result_available_after}ms")

        return results

    def add_concept(self, node: Concept):
        raise NotImplementedError("Ingestion in Neo4j is not yet implemented")

    def add_triple(self, triple: Triple):
        raise NotImplementedError("Ingestion in Neo4j is not yet implemented")

    def _get_concept(self, node_identifier: ConceptIdentifier) -> Concept:
        query = (
            f"MATCH (n {{node_id: '{node_identifier}'}})"
            " RETURN n.node_id AS node_id, n.node_name AS node_name, "
            "     LABELS(n)[0] AS domain_identifier"
        )
        logger.info("Retrieving the node from Neo4j")
        logger.info(f"Query: {query}")
        start = time.time()
        data = self._execute_query(query)
        end = time.time()
        logger.info(f"Neo4j query success after {end - start:.2f} seconds.")

        for row in data:
            return Concept(
                identifier=row["node_id"],
                label=row["node_name"],
                domain_identifier=row["domain_identifier"],
            )

        raise ValueError(f"Node with identifier {node_identifier} not found in Neo4j KG")

    def get_concepts_from_domain(
        self,
        domain_identifier: DomainIdentifier,
    ) -> list[Concept]:
        query = (
            f"MATCH (n:{domain_identifier}) RETURN n.node_id AS node_id, n.node_name as node_name"
        )
        logger.info("Retrieving concepts from Neo4j. It will take some time.")
        logger.info(f"Query: {query}")
        start = time.time()
        data = self._execute_query(query)
        end = time.time()
        logger.info(f"Neo4j query success after {end - start:.2f} seconds.")

        concepts = []
        for row in data:
            concepts.append(
                Concept(
                    identifier=row["node_id"],
                    label=row["node_name"],
                    domain_identifier=domain_identifier,
                )
            )
        return concepts

    def _get_triples(
        self,
        subject_identifier: ConceptIdentifier | None = None,
        object_identifier: ConceptIdentifier | None = None,
        relation_identifier: RelationIdentifier | None = None,
    ) -> list[Triple]:
        s_pattern = (
            "(s)" if subject_identifier is None else f"(s {{node_id: '{subject_identifier}'}})"
        )
        r_pattern = "[r]" if relation_identifier is None else f"[r:`{relation_identifier}`]"
        o_pattern = (
            "(o)" if object_identifier is None else f"(o {{node_id: '{object_identifier}'}})"
        )
        query = (
            "MATCH "
            + s_pattern
            + "-"
            + r_pattern
            + "->"
            + o_pattern
            + " RETURN s.node_id AS s_node_id, TYPE(r) AS r_identifier, o.node_id AS o_node_id"
        )
        logger.info("Retrieving triples from Neo4j. It will take some time.")
        logger.info(f"Query: {query}")
        start = time.time()

        start = time.time()
        data = self._execute_query(query)
        end = time.time()
        logger.info(f"Neo4j query success after {end - start:.2f} seconds.")
        triples = []
        for row in data:
            triples.append(
                Triple(
                    subject_identifier=row["s_node_id"],
                    relation_identifier=row["r_identifier"],
                    object_identifier=row["o_node_id"],
                )
            )
        return triples
