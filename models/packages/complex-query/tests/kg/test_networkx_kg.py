import pytest
from query_common.entities.kg.concept import Concept
from query_common.entities.kg.triple import Triple

from complex_query.core.entities.config.kg import NetworkxKGConfig
from complex_query.impl.kg.networkx_kg import NetworkxKG

# Concept, relation, domain IDs
CID1, CID2, CID3, CID4 = "1", "cid2", "3", "4"
DID1, DID2 = "D1", "D2"
RID1, RID2 = "IS_A", "REL_7"


@pytest.fixture
def kg():
    config = NetworkxKGConfig()
    nx_kg = NetworkxKG(config)

    nx_kg.add_concept(Concept(identifier=CID1, label="Node1", domain_identifier=DID1))
    nx_kg.add_concept(Concept(identifier=CID2, label="Node2", domain_identifier=DID2))
    nx_kg.add_concept(Concept(identifier=CID3, label="Node3", domain_identifier=DID1))
    nx_kg.add_concept(Concept(identifier=CID4, label="Node4", domain_identifier=DID2))
    nx_kg.add_triple(
        Triple(subject_identifier=CID1, relation_identifier=RID1, object_identifier=CID2)
    )
    nx_kg.add_triple(
        Triple(subject_identifier=CID2, relation_identifier=RID1, object_identifier=CID1)
    )
    return nx_kg


def test_add_and_get_concept(kg):
    concept = kg.get_concept(CID1)
    assert concept.label == "Node1"
    assert concept.identifier == CID1


def test_get_triples_by_subject(kg):
    triples = kg.get_triples(subject_identifier=CID1)
    assert len(triples) == 1
    assert triples[0].object_identifier == CID2


def test_get_triples_by_object(kg):
    triples = kg.get_triples(object_identifier=CID1)
    assert len(triples) == 1
    assert triples[0].subject_identifier == CID2


def test_get_triples_by_relation(kg):
    triples = kg.get_triples(relation_identifier=RID1)
    assert len(triples) == 2
    assert all(triple.relation_identifier == RID1 for triple in triples)


def test_get_concepts_by_domain(kg):
    concepts = kg.get_concepts_from_domain(domain_identifier=DID1)
    assert len(concepts) == 2
    assert all(concept.domain_identifier == DID1 for concept in concepts)
