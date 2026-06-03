import logging

import pytest
from dependency_injector import providers
from dependency_injector.containers import DynamicContainer
from query_common.entities.kg.concept import Concept

from complex_query.core.entities.config.kg import Neo4jKGConfig
from complex_query.impl.kg.neo4j_kg import Neo4jKG


@pytest.fixture(scope="module")
def kg():
    config = Neo4jKGConfig(use_okta=True)
    return Neo4jKG(config)


@pytest.fixture
def container():
    container = DynamicContainer()
    container.logger = providers.Object(logging.getLogger())
    container.init_resources()
    container.wire(packages=["complex_query"])
    yield container
    container.unwire()


@pytest.mark.neo4j
@pytest.mark.parametrize(
    "cid", ["0009ad04c8987619d66a00dceb438645", "001c6cfebaba7d7927b0a197195fb7ac"]
)
def test_get_node(kg, cid):
    result = kg.get_node(cid)

    assert isinstance(result, Concept)
    assert result.identifier == cid


@pytest.mark.neo4j
def test_get_concepts_from_domain(kg):
    concepts = kg.get_concepts_from_domain("GENE")
    assert len(concepts) > 0


@pytest.mark.neo4j
@pytest.mark.parametrize(
    ["s", "r", "o"],
    [
        ("0003fb22e2049da4ab2c993efa06f726", "ASSOCIATE", "2a57ef5e287a4d2b1bc3cb68a39bf1f9"),
        ("0003fb22e2049da4ab2c993efa06f726", "ASSOCIATE", None),
        ("0003fb22e2049da4ab2c993efa06f726", None, "2a57ef5e287a4d2b1bc3cb68a39bf1f9"),
        (None, "ASSOCIATE", "2a57ef5e287a4d2b1bc3cb68a39bf1f9"),
        (None, "ASSOCIATE", "2a57ef5e287a4d2b1bc3cb68a39bf1f9"),
        ("0003fb22e2049da4ab2c993efa06f726", None, None),
    ],
)
def test_get_triples(kg, s, r, o):
    triples = kg.get_triples(subject_identifier=s, relation_identifier=r, object_identifier=o)
    assert len(triples) > 0
    if s is not None:
        assert s in [t.subject_identifier for t in triples]
    if r is not None:
        assert r in [t.relation_identifier for t in triples]
    if o is not None:
        assert o in [t.object_identifier for t in triples]
