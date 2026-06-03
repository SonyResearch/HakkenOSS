from __future__ import annotations

from typing import TYPE_CHECKING, Any

import networkx as nx
from loguru import logger
from query_common.entities.kg.concept import Concept
from query_common.entities.kg.triple import Triple

from complex_query.core.contracts.kg import KnowledgeGraph
from complex_query.core.entities.config.kg import NetworkxKGConfig

if TYPE_CHECKING:
    from networkx.classes.reportviews import (
        InMultiEdgeDataView,
        OutMultiEdgeDataView,
        OutMultiEdgeView,
    )
    from query_common.entities.kg.identifier import (
        ConceptIdentifier,
        DomainIdentifier,
        RelationIdentifier,
    )


def _nx_edge_view_to_triples(
    nx_edges: (
        OutMultiEdgeView[ConceptIdentifier]
        | OutMultiEdgeDataView[ConceptIdentifier, Any]
        | InMultiEdgeDataView[ConceptIdentifier, Any]
    ),
) -> list[Triple]:
    return [
        Triple(
            subject_identifier=subject_identifier,
            relation_identifier=relation_identifier,
            object_identifier=object_identifier,
        )
        for (subject_identifier, object_identifier, relation_identifier) in nx_edges
    ]


class NetworkxKG(KnowledgeGraph[NetworkxKGConfig]):
    """Simple in-memory KG."""

    def __init__(self, config: NetworkxKGConfig) -> None:
        super().__init__(config)

        self.graph: nx.MultiDiGraph[ConceptIdentifier] = nx.MultiDiGraph()

    def add_concept(self, node: Concept) -> None:
        self.graph.add_node(
            node.identifier,
            identifier=node.identifier,
            label=node.label,
            domain_identifier=node.domain_identifier,
        )

    def _get_concept(self, node_identifier: ConceptIdentifier) -> Concept:
        try:
            return self.node_data_to_concept(self.graph.nodes[node_identifier])
        except KeyError as e:
            logger.exception(f"Could not find concept {node_identifier} in the graph.")
            raise ValueError(f"Could not find concept {node_identifier} in the graph.") from e

    def add_triple(self, triple: Triple) -> None:
        """The relation ID is stored as an edge key"""
        self.graph.add_edge(
            triple.subject_identifier, triple.object_identifier, key=triple.relation_identifier
        )

    def _get_triples(
        self,
        subject_identifier: ConceptIdentifier | None = None,
        object_identifier: ConceptIdentifier | None = None,
        relation_identifier: RelationIdentifier | None = None,
    ) -> list[Triple]:
        """
        Returns triples in the graph that match the query.
        """
        if (
            subject_identifier is not None
            and object_identifier is not None
            and relation_identifier is not None
        ):
            if self.graph.has_edge(subject_identifier, object_identifier, relation_identifier):
                return [
                    Triple(
                        subject_identifier=subject_identifier,
                        object_identifier=object_identifier,
                        relation_identifier=relation_identifier,
                    )
                ]
            return []

        if subject_identifier is None and object_identifier is None and relation_identifier is None:
            edges = self.graph.edges
        elif subject_identifier is not None:
            edges = self.graph.out_edges(subject_identifier, data=False, keys=True)  # type: ignore
            if object_identifier is not None:
                edges = [edge for edge in edges if edge[1] == object_identifier]  # type: ignore
        else:
            edges = self.graph.in_edges(object_identifier, data=False, keys=True)  # type: ignore
        triples = _nx_edge_view_to_triples(edges)
        if relation_identifier is not None:
            triples = [t for t in triples if t.relation_identifier == relation_identifier]
        return triples

    def get_concepts_from_domain(self, domain_identifier: DomainIdentifier) -> list[Concept]:
        nodes_data = [
            n_data
            for (_, n_data) in self.graph.nodes(data=True)
            if n_data["domain_identifier"] == domain_identifier
        ]
        return [self.node_data_to_concept(n_data) for n_data in nodes_data]

    @staticmethod
    def node_data_to_concept(node_data: dict[str, Any]) -> Concept:
        return Concept(
            identifier=node_data["identifier"],
            label=node_data["label"],
            domain_identifier=node_data["domain_identifier"],
        )
