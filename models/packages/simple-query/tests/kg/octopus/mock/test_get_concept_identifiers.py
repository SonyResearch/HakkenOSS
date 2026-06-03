import re
from typing import TYPE_CHECKING, Any

import pytest

from simple_query.kg.impl.neo4j import Neo4jKnowledgeGraph
from simple_query.kg.values.errors import Neo4jKGError
from simple_query.query.entities.inputs import (
    Argument,
    ConditionNode,
    ConditionPredicate,
)
from simple_query.query.values.types import ConditionType

if TYPE_CHECKING:
    from query_common.entities.kg.identifier import DomainIdentifier


class MockNeo4jKnowledgeGraph(Neo4jKnowledgeGraph):
    def __init__(self):
        self.client = None

    def _execute_neo4j_query(self, neo4j_query: str) -> list[dict[str, Any]]:
        # WHERE condition is assumed to be one of
        # - NOT P(?, r1, n2)
        # - P(?, r2, n1)
        # - P(n4, r3, ?)

        # Graph data
        # - (n1, r1, n2)  # noqa: ERA001
        # - (n3, r2, n1)  # noqa: ERA001
        # - (n2, r2, n1)  # noqa: ERA001
        # - (n4, r3, n2)  # noqa: ERA001

        node_domain_mapping = {
            "n1": "GENE",
            "n2": "CHEMICAL",
            "n3": "GENE",
            "n4": "CHEMICAL",
        }

        domain_identifier: DomainIdentifier | None = None

        if match := re.search(r"MATCH \(n:(\S+)\)", neo4j_query):
            # When no condition is given; return all node IDs of domain
            domain_identifier = match.group(1)
            return [
                {"node_id": k} for k, v in node_domain_mapping.items() if v == domain_identifier
            ]

        if match := re.search(r"'(\S+)' IN LABELS\(", neo4j_query):
            domain_identifier = match.group(1)

        print(neo4j_query)
        if "o.node_id = 'n2'" in neo4j_query:
            # NOT P(?, r1, n2)
            node_ids = ["n3", "n2", "n4"]
        elif "o.node_id = 'n1'" in neo4j_query:
            # P(?, r2, n1)
            node_ids = ["n3", "n2"]
        elif "s.node_id = 'n4'" in neo4j_query:
            # P(n4, r3, ?)
            node_ids = ["n2"]
        else:
            raise ValueError(f"unable to recognize the query: {neo4j_query}")

        if domain_identifier is not None:
            node_ids = [
                node_id for node_id in node_ids if node_domain_mapping[node_id] == domain_identifier
            ]

        return [{"node_id": node_id} for node_id in node_ids]


@pytest.fixture
def kg() -> Neo4jKnowledgeGraph:
    return MockNeo4jKnowledgeGraph()


@pytest.fixture
def condition() -> ConditionNode:
    # NOT EXISTS(X, r1, n2) AND (EXISTS(X, r2, n1) OR EXISTS(n4, r3, X))
    x_r1_n2 = ConditionNode(
        type=ConditionType.LEAF,
        predicate=ConditionPredicate(
            subject=Argument(value="X", is_variable=True),
            relation=Argument(value="r1"),
            object=Argument(value="n2"),
        ),
    )
    not_x_r1_n2 = ConditionNode(type=ConditionType.NOT, children=[x_r1_n2])
    x_r2_n1 = ConditionNode(
        type=ConditionType.LEAF,
        predicate=ConditionPredicate(
            subject=Argument(value="X", is_variable=True),
            relation=Argument(value="r2"),
            object=Argument(value="n1"),
        ),
    )
    n4_r3_x = ConditionNode(
        type=ConditionType.LEAF,
        predicate=ConditionPredicate(
            subject=Argument(value="n4"),
            relation=Argument(value="r3"),
            object=Argument(value="X", is_variable=True),
        ),
    )

    return ConditionNode(
        type=ConditionType.AND,
        children=[
            not_x_r1_n2,
            ConditionNode(type=ConditionType.OR, children=[x_r2_n1, n4_r3_x]),
        ],
    )


@pytest.mark.parametrize(
    "domain_identifier,expected",
    [(None, {"n2", "n3"}), ("GENE", {"n3"}), ("CHEMICAL", {"n2"})],
)
def test_get_concept_identifier_set_for_condition(kg, condition, domain_identifier, expected):
    assert (
        kg._get_concept_identifier_set_for_condition(
            condition=condition, domain_identifier=domain_identifier
        )
        == expected
    )


def test_get_concept_identifiers_with_condition(kg, condition):
    concept_identifiers = kg.get_concept_identifiers(condition=condition, domain_identifier=None)
    assert isinstance(concept_identifiers, list)
    assert isinstance(concept_identifiers[0], str)
    assert set(concept_identifiers) == {"n2", "n3"}


@pytest.mark.parametrize(
    "domain_identifier,expected",
    [("GENE", {"n1", "n3"}), ("CHEMICAL", {"n2", "n4"})],
)
def test_get_concept_identifiers_without_condition(kg, domain_identifier, expected):
    concept_identifiers = kg.get_concept_identifiers(
        domain_identifier=domain_identifier, condition=None
    )
    assert set(concept_identifiers) == expected


def test_get_concept_node_ids_without_condition_and_domain(kg):
    with pytest.raises(Neo4jKGError):
        kg.get_concept_identifiers(domain_identifier=None, condition=None)
