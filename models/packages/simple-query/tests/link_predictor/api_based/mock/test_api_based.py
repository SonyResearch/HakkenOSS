import pytest
import responses

from simple_query.link_predictor.entities.configs import ApiBasedLinkPredictorConfig
from simple_query.link_predictor.entities.inputs import LinkPredictorInputTriple
from simple_query.link_predictor.impl.api_based import ApiBasedLinkPredictor
from simple_query.link_predictor.values.errors import LinkPredictorError


@pytest.fixture
def link_predictor() -> ApiBasedLinkPredictor:
    return ApiBasedLinkPredictor(config=ApiBasedLinkPredictorConfig(url="http://test"))


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


@pytest.fixture
def relation_probabilities() -> list[list[float]]:
    return [[0.4, 0.6], [0.3, 0.7]]


@responses.activate
def test_predict(link_predictor, triples, relation_probabilities):
    response = responses.Response(
        method="POST",
        url=link_predictor.config.url,
        json={"relations_ids": ["INDUCES", "INHIBITS"], "relations_probs": relation_probabilities},
    )
    responses.add(response)

    assert link_predictor.predict(triples) == [0.4, 0.7]


@responses.activate
def test_predict_with_500(link_predictor, triples):
    response = responses.Response(method="POST", url=link_predictor.config.url, status=500)
    responses.add(response)

    with pytest.raises(LinkPredictorError, match=r".*status code 500.*"):
        link_predictor.predict(triples)


@responses.activate
def test_predict_with_invalid_key(link_predictor, triples):
    response = responses.Response(
        method="POST", url=link_predictor.config.url, json={"invalid_key": "value"}
    )
    responses.add(response)

    with pytest.raises(LinkPredictorError, match=r".*does not have the key.*"):
        link_predictor.predict(triples)


def test_predict_with_empty_input():
    # For empty input, no API call will be made,
    # so it must pass the test even if we give a wrong URL.
    link_predictor = ApiBasedLinkPredictor(
        config=ApiBasedLinkPredictorConfig(url="http://fake_url")
    )
    result = link_predictor.predict(triples=[])
    assert isinstance(result, list)
    assert not result
