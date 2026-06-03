import pytest

from filtering.core.entities.kg import EdgeDirection, YearRange
from filtering.impl.kg.neo4j_kg import Neo4jKnowledgeGraph


@pytest.mark.neo4j
class TestNeo4jGraph:
    def test(self, neo4j_kg_config):
        kg = Neo4jKnowledgeGraph(neo4j_kg_config)

        node_ids = [
            "0009ad04c8987619d66a00dceb438645",
            "001c6cfebaba7d7927b0a197195fb7ac",
            "0033a95d509b8116c1318d23fd986aa6",
            "_NOT_EXISTS_",
        ]

        all_degrees = kg.get_degrees(node_ids, direction=EdgeDirection.ALL)
        in_degrees = kg.get_degrees(node_ids, direction=EdgeDirection.IN)
        out_degrees = kg.get_degrees(node_ids, direction=EdgeDirection.OUT)
        for i in range(len(in_degrees)):
            assert in_degrees[i] + out_degrees[i] == all_degrees[i]

        in_degrees = kg.get_degrees(
            node_ids, direction=EdgeDirection.IN, year_range=YearRange(2010, 2020)
        )
        out_degrees = kg.get_degrees(
            node_ids, direction=EdgeDirection.OUT, year_range=YearRange(2010, 2020)
        )
        all_degrees = kg.get_degrees(
            node_ids, direction=EdgeDirection.ALL, year_range=YearRange(2010, 2020)
        )
        for i in range(len(in_degrees)):
            assert in_degrees[i] + out_degrees[i] == all_degrees[i]

        kg.get_degrees(
            node_ids,
            direction=EdgeDirection.IN,
            year_range=YearRange(2010, 2020),
        )
        kg.get_degrees(
            node_ids,
            direction=EdgeDirection.IN,
            year_range=YearRange(2010, 2020),
        )
