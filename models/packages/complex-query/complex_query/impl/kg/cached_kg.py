import time
from typing import TYPE_CHECKING

from loguru import logger

from complex_query.core.contracts.kg import KnowledgeGraph

if TYPE_CHECKING:
    from query_common.entities.kg.concept import Concept
    from query_common.entities.kg.identifier import (
        ConceptIdentifier,
        DomainIdentifier,
        RelationIdentifier,
    )
    from query_common.entities.kg.triple import Triple

    from complex_query.core.contracts.kg_ledger import KnowledgeGraphLedger


class CachedKnowledgeGraph(KnowledgeGraph):
    """
    CachedKnowledgeGraph combines a KnowledgeGraph (e.g. Neo4j) with a KnowledgeGraphLedger
    (e.g. HDF5) to simulate a caching layer.

    It attempts to retrieve information from the ledger first. If the information is not found,
    it uses the main KnowledgeGraph and updates the ledger with the new information.
    This is to improve performance for repeated queries.
    """

    def __init__(
        self,
        base_kg: KnowledgeGraph,
        ledger: "KnowledgeGraphLedger",
    ):
        self.base_kg = base_kg
        self.ledger = ledger
        logger.info(f"Initialized CachedKnowledgeGraph with {base_kg} and {ledger}")

    def add_concept(
        self,
        node: "Concept",
    ) -> None:
        logger.info(f"Adding node {node.identifier} to KnowledgeGraph and Ledger")
        self.base_kg.add_concept(node)
        self.ledger.add_concept(node)

    def add_triple(
        self,
        triple: "Triple",
    ) -> None:
        logger.info(f"Adding triple {triple} to KnowledgeGraph and Ledger")
        self.base_kg.add_triple(triple)
        self.ledger.add_triple(triple)

    def _get_concept(
        self,
        node_identifier: "ConceptIdentifier",
    ) -> "Concept":
        logger.info(f"Attempting to get node {node_identifier} from Ledger")
        try:
            node = self.ledger.get_concept(node_identifier)
            logger.info(f"Node {node_identifier} found in Ledger")
            return node
        except KeyError:
            logger.info(f"Node {node_identifier} not found in Ledger, fetching from KnowledgeGraph")
            node = self.base_kg._get_concept(node_identifier)
            self.ledger.add_concept(node)
            logger.info(f"Node {node_identifier} added to Ledger")
            return node

    def get_concepts_from_domain(self, domain_identifier: "DomainIdentifier") -> list["Concept"]:
        logger.info(f"Attempting to get concepts from domain {domain_identifier} from ledger")
        try:
            start = time.time()
            concepts = self.ledger.get_concepts_from_domain(domain_identifier)
            logger.info(
                f"Retrieved {len(concepts)} concepts for domain {domain_identifier} in ledger in "
                f"{time.time() - start:.2f} seconds."
            )
            return concepts
        except KeyError:
            logger.info(
                f"Domain {domain_identifier} not found or not complete in ledger, "
                "fetching from KnowledgeGraph"
            )
            concepts = self.base_kg.get_concepts_from_domain(domain_identifier)
            if concepts:  # Only add to ledger if the domain exists in the main KG
                start = time.time()
                self.ledger.add_concepts_for_domain(
                    concepts=concepts, domain_identifier=domain_identifier
                )
                logger.info(
                    f"Added {len(concepts)} concepts for domain {domain_identifier} to ledger in "
                    f"{time.time() - start:.2f} seconds."
                )
            else:
                logger.info(f"Domain {domain_identifier} not found in KnowledgeGraph")
            return concepts

    def _get_triples(
        self,
        subject_identifier: "ConceptIdentifier | None" = None,
        object_identifier: "ConceptIdentifier | None" = None,
        relation_identifier: "RelationIdentifier | None" = None,
    ) -> list["Triple"]:
        logger.info(
            "Attempting to get triples from Ledger with params: "
            f"subject_identifier={subject_identifier}, "
            f"object_identifier={object_identifier}, "
            f"relation_identifier={relation_identifier}"
        )
        triples = self.ledger.get_triples(
            subject_identifier, object_identifier, relation_identifier
        )
        if triples:
            logger.info(f"Found {len(triples)} triples in Ledger")
            return triples

        logger.info("No triples found in Ledger, fetching from KnowledgeGraph")
        triples = self.base_kg._get_triples(
            subject_identifier, object_identifier, relation_identifier
        )
        for triple in triples:
            self.ledger.add_triple(triple)
        logger.info(f"Added {len(triples)} triples to Ledger")
        return triples
