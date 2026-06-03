import pytest
from query_common.entities.kg.triple import Triple

from complex_query.core.entities.config.link_predictor import RandomLinkPredictorConfig
from complex_query.impl.link_predictor import RandomLinkPredictor


@pytest.fixture
def link_predictor() -> RandomLinkPredictor:
    return RandomLinkPredictor(config=RandomLinkPredictorConfig())


@pytest.fixture
def triples() -> list[Triple]:
    return [
        Triple(subject_identifier="n2", relation_identifier="R1", object_identifier="n3"),
        Triple(subject_identifier="n1", relation_identifier="R1", object_identifier="n2"),
    ]


def test_predict(link_predictor, triples):
    preds_list = []

    for i in range(2):
        preds_list.append([])
        for triple in triples:
            preds_list[i].append(link_predictor.predict(triple))

    assert preds_list[0] != preds_list[1]
    for preds in preds_list:
        for pred in preds:
            assert 0 <= pred <= 1
