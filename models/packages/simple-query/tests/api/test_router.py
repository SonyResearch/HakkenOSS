from unittest.mock import patch

import pytest
from dotenv import load_dotenv
from query_common.entities.kg.concept import Concept
from query_common.entities.kg.relation import Relation
from query_common.entities.query import Candidate, QueryRequest, RequestVariable

from simple_query.api.container import ApiConfig, SimpleQueryingContainer
from simple_query.api.entities import ConstraintFilteringRequest
from simple_query.api.router import filter_constraint, query
from simple_query.kg.entities.constraint import (
    ConstraintFilteringOutput,
    ConstraintFilteringOutputEntry,
)
from simple_query.kg.impl.neo4j import Neo4jKnowledgeGraph
from simple_query.query.impl.simple import SimpleQuerying


@pytest.fixture
def container(envfile_path) -> SimpleQueryingContainer:
    load_dotenv(envfile_path, override=True)
    api_config = ApiConfig()  # type: ignore
    container = SimpleQueryingContainer()
    container.config.from_pydantic(api_config)
    return container


@pytest.fixture
def query_request() -> QueryRequest:
    return QueryRequest(
        formula="P(X, r1, n3) AND EXISTS(n4, r2, X)",
        variables=[RequestVariable(label="X", domain="GENE")],
        n_candidates=10,
    )


@pytest.fixture
def candidates() -> list[Candidate]:
    return [
        Candidate(var_assignments={"X": "n1"}, condition_scores={}, query_score=0.8),
        Candidate(var_assignments={"X": "n2"}, condition_scores={}, query_score=0.6),
    ]


@pytest.fixture
def constraint_filtering_request() -> ConstraintFilteringRequest:
    return ConstraintFilteringRequest.model_validate(
        {  # Both domain and domain_identifier are accepted
            "subject": {"value": "n1", "domain": "CHEMICAL"},
            "relation": {"value": "R", "is_variable": True},
            "object": {"value": "X", "domain_identifier": "GENE", "is_variable": True},
        }
    )


@pytest.fixture
def constraint_filtering_output() -> ConstraintFilteringOutput:
    return [
        ConstraintFilteringOutputEntry(
            variable="R", type="relation", values=[Relation(identifier="relation_1")]
        ),
        ConstraintFilteringOutputEntry(
            variable="X",
            type="concept",
            values=[
                Concept(identifier="c1", label="concept_1", domain_identifier="GENE"),
                Concept(identifier="c2", label="concept_2", domain_identifier="GENE"),
            ],
        ),
    ]


def test_query(container, query_request, candidates):
    container.wire(modules=[__name__], packages=["simple_query.api.router"])
    with patch.object(SimpleQuerying, "find_candidates", return_value=candidates):
        response = query(request=query_request)
    assert len(response.candidates) == 2


def test_filter_constraint(container, constraint_filtering_request, constraint_filtering_output):
    container.wire(modules=[__name__], packages=["simple_query.api.router"])
    assert constraint_filtering_request.subject.domain_identifier == "CHEMICAL"
    assert constraint_filtering_request.object.domain_identifier == "GENE"
    with patch.object(
        Neo4jKnowledgeGraph, "filter_constraint", return_value=constraint_filtering_output
    ):
        response = filter_constraint(constraint_filtering_request)
    assert len(response) == 2
