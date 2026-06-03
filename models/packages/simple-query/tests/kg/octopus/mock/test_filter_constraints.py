from typing import Any

import pytest

from simple_query.kg.entities.constraint import TripleConstraint, TripleConstraintArgument
from simple_query.kg.impl.neo4j import Neo4jKnowledgeGraph
from simple_query.kg.values.errors import Neo4jKGError


class MockNeo4jKnowledgeGraph(Neo4jKnowledgeGraph):
    def __init__(self):
        self.client = None

    def _execute_neo4j_query(self, neo4j_query: str) -> list[dict[str, Any]]:  # noqa: ARG002
        return [
            {
                "X": [
                    {"node_id": "n1", "node_name": "node_1", "domain_identifier": "GENE"},
                    {"node_id": "n2", "node_name": "node_2", "domain_identifier": "CHEMICAL"},
                ],
                "R": [
                    {"relation_type": "relation_1"},
                    {"relation_type": "relation_2"},
                    {"relation_type": "relation_3"},
                ],
            }
        ]


@pytest.fixture
def kg() -> Neo4jKnowledgeGraph:
    return MockNeo4jKnowledgeGraph()


@pytest.fixture
def triple_constraint() -> TripleConstraint:
    return TripleConstraint(
        subject=TripleConstraintArgument(value="n3", domain_identifier="GENE"),
        relation=TripleConstraintArgument(value="R", is_variable=True),
        object=TripleConstraintArgument(value="X", is_variable=True),
    )


@pytest.fixture
def erroneous_triple_constraint() -> TripleConstraint:
    return TripleConstraint(
        subject=TripleConstraintArgument(value="n3", domain_identifier="GENE"),
        relation=TripleConstraintArgument(
            value="R", is_variable=True, domain_identifier="CHEMICAL"
        ),
        object=TripleConstraintArgument(value="X", is_variable=True, domain_identifier="CHEMICAL"),
    )


def test_filter_constraint(kg, triple_constraint):
    output = kg.filter_constraint(triple_constraint)
    assert len(output) == 2
    for output_entry in output:
        if output_entry.variable == "X":
            assert output_entry.type == "concept"
            assert len(output_entry.values) == 2
        elif output_entry.variable == "R":
            assert output_entry.type == "relation"
            assert len(output_entry.values) == 3
        else:
            raise AssertionError()


def test_filter_constraint_errors(kg, erroneous_triple_constraint):
    with pytest.raises(Neo4jKGError):
        kg.filter_constraint(erroneous_triple_constraint)
