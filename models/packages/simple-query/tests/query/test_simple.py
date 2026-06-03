from unittest.mock import Mock

import pytest
from query_common.entities.query import Candidate

from simple_query.kg.base import KnowledgeGraph
from simple_query.link_predictor.base import LinkPredictor
from simple_query.query.entities.inputs import (
    Argument,
    ConditionNode,
    ConditionPredicate,
    QueryInput,
    TargetPredicate,
)
from simple_query.query.impl.simple import SimpleQuerying, SimpleQueryingConfig
from simple_query.query.values.types import ConditionType


@pytest.fixture
def kg() -> KnowledgeGraph:
    mock_kg = Mock(spec=KnowledgeGraph)
    mock_kg.get_concept_identifiers.return_value = ["n1", "n2"]
    return mock_kg


@pytest.fixture
def link_predictor() -> LinkPredictor:
    mock_link_predictor = Mock(spec=LinkPredictor)
    mock_link_predictor.predict.return_value = [0.4, 0.6]
    return mock_link_predictor


@pytest.fixture
def querying(kg, link_predictor) -> SimpleQuerying:
    return SimpleQuerying(config=SimpleQueryingConfig(), kg=kg, link_predictor=link_predictor)


@pytest.fixture
def query_input() -> QueryInput:
    return QueryInput(
        target_predicate=TargetPredicate(
            name="P",
            subject=Argument(value="X", is_variable=True),
            relation=Argument(value="r1"),
            object=Argument(value="n3"),
        ),
        variable_name="X",
        variable_domain_identifier="GENE",
        condition=ConditionNode(
            type=ConditionType.LEAF,
            predicate=ConditionPredicate(
                subject=Argument(value="n4"),
                relation=Argument(value="r2"),
                object=Argument(value="X", is_variable=True),
            ),
        ),
    )


@pytest.fixture
def query_input_without_condition() -> QueryInput:
    return QueryInput(
        target_predicate=TargetPredicate(
            name="P",
            subject=Argument(value="X", is_variable=True),
            relation=Argument(value="r1"),
            object=Argument(value="n3"),
        ),
        variable_name="X",
        variable_domain_identifier="GENE",
        condition=None,
    )


def test_find_candidates(querying, query_input):
    expected = [
        Candidate(var_assignments={"X": "n2"}, condition_scores={}, query_score=0.6),
        Candidate(var_assignments={"X": "n1"}, condition_scores={}, query_score=0.4),
    ]
    assert querying.find_candidates(query_input) == expected


def test_find_candidates_without_condition(querying, query_input):
    expected = [
        Candidate(var_assignments={"X": "n2"}, condition_scores={}, query_score=0.6),
        Candidate(var_assignments={"X": "n1"}, condition_scores={}, query_score=0.4),
    ]
    assert querying.find_candidates(query_input) == expected
