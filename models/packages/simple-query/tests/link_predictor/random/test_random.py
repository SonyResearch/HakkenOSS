import pytest

from simple_query.link_predictor.entities.configs import (
    RandomLinkPredictorConfig,
)
from simple_query.link_predictor.entities.inputs import LinkPredictorInputTriple
from simple_query.link_predictor.impl.random import RandomLinkPredictor


@pytest.fixture
def link_predictor() -> RandomLinkPredictor:
    return RandomLinkPredictor(config=RandomLinkPredictorConfig(seed=123))


@pytest.fixture
def triples() -> list[LinkPredictorInputTriple]:
    return [
        LinkPredictorInputTriple(
            subject_identifier="n2", relation_identifier="INDUCES", object_identifier="n3"
        ),
        LinkPredictorInputTriple(
            subject_identifier="n1", relation_identifier="INHIBITS", object_identifier="n2"
        ),
    ]


def test_predict(link_predictor, triples):
    values = link_predictor.predict(triples)

    assert len(values) == 2
    assert all(0 <= v <= 1.0 for v in values)
