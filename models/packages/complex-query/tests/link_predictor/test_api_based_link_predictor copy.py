from typing import Any

import pytest
import responses
from query_common.entities.kg.triple import Triple

from complex_query.core.entities.config.link_predictor import ApiBasedLinkPredictorConfig
from complex_query.impl.link_predictor import ApiBasedLinkPredictor


@pytest.fixture
def link_predictor() -> ApiBasedLinkPredictor:
    return ApiBasedLinkPredictor(config=ApiBasedLinkPredictorConfig(url="http://test"))


@pytest.fixture
def triples_batch_single_relation() -> list[Triple]:
    return [
        Triple(subject_identifier="n2", relation_identifier="R1", object_identifier="n3"),
        Triple(subject_identifier="n1", relation_identifier="R1", object_identifier="n2"),
    ]


@pytest.fixture
def output_single_relation() -> list[dict[str, Any]]:
    return [
        {"relations_ids": ["R1"], "relations_probs": [[0.4]], "relations_scores": [[0.1]]},
        {"relations_ids": ["R1"], "relations_probs": [[0.6]], "relations_scores": [[0.2]]},
    ]


@pytest.fixture
def output_batch_single_relation() -> dict[str, Any]:
    return {
        "relations_ids": ["R1"],
        "relations_probs": [[0.4], [0.6]],
        "relations_scores": [[0.1, 0.2]],
    }


@pytest.fixture
def triples_batch_multi_relation() -> list[Triple]:
    return [
        Triple(subject_identifier="n2", relation_identifier="R1", object_identifier="n3"),
        Triple(subject_identifier="n1", relation_identifier="R2", object_identifier="n2"),
    ]


@pytest.fixture
def output_multi_relation() -> list[dict[str, Any]]:
    return [
        {"relations_ids": ["R1"], "relations_probs": [[0.4]], "relations_scores": [[0.1]]},
        {"relations_ids": ["R2"], "relations_probs": [[0.7]], "relations_scores": [[-0.1]]},
    ]


@pytest.fixture
def output_batch_multi_relation() -> dict[str, Any]:
    return {
        "relations_ids": ["R1", "R2"],
        "relations_probs": [[0.4, 0.6], [0.3, 0.7]],
        "relations_scores": [[0.1, 0.2], [-0.5, -0.1]],
    }


@responses.activate
def test_predict_single_relation(
    link_predictor,
    triples_batch_single_relation,
    output_single_relation,
    output_batch_single_relation,
):
    predicted = []
    for i, triple in enumerate(triples_batch_single_relation):
        output = output_single_relation[i]
        with responses.RequestsMock() as rsps:
            rsps.add(method="POST", url=link_predictor.config.url, json=output)
            score = link_predictor.predict(triple)
        predicted.append(score)

    with responses.RequestsMock() as rsps:
        rsps.add(method="POST", url=link_predictor.config.url, json=output_batch_single_relation)
        predicted_batch = link_predictor.predict_batch(triples_batch_single_relation)

    answer_probs = [0.4, 0.6]

    assert predicted == predicted_batch.tolist() == answer_probs


@responses.activate
def test_predict_multi_relation(
    link_predictor,
    triples_batch_multi_relation,
    output_multi_relation,
    output_batch_multi_relation,
):
    predicted = []
    for i, triple in enumerate(triples_batch_multi_relation):
        output = output_multi_relation[i]
        with responses.RequestsMock() as rsps:
            rsps.add(method="POST", url=link_predictor.config.url, json=output)
            score = link_predictor.predict(triple)
        predicted.append(score)

    with responses.RequestsMock() as rsps:
        rsps.add(method="POST", url=link_predictor.config.url, json=output_batch_multi_relation)
        predicted_batch = link_predictor.predict_batch(triples_batch_multi_relation)

    answer_probs = [0.4, 0.7]
    assert predicted == predicted_batch.tolist() == answer_probs
