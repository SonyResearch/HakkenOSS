from query_common.entities.kg.concept import Concept
from query_common.entities.kg.relation import Relation
from query_common.entities.kg.triple import Triple


def test_concept_creation():
    concept = Concept(identifier="a123", label="Test Concept")
    assert concept.identifier == "a123"
    concept = Concept(identifier="a123", label="Test Concept")
    assert concept.identifier == "a123"
    assert concept.label == "Test Concept"

    concept = Concept(identifier="123", label="Test Concept")
    assert concept.identifier == "123"
    concept = Concept(identifier="123", label="Test Concept")
    assert concept.identifier == "123"
    assert concept.label == "Test Concept"


def test_relation_creation():
    relation = Relation(identifier="relation_a", label="Test Relation")
    assert relation.identifier == "relation_a"
    relation = Relation(identifier="relation_a", label="Test Relation")
    assert relation.identifier == "relation_a"
    assert relation.label == "Test Relation"

    relation = Relation(identifier="relation_a")
    assert relation.identifier == "relation_a"
    assert relation.label == "relation_a"


def test_triple_creation():
    triple = Triple(subject_identifier="s", relation_identifier="r", object_identifier="o")
    assert triple.subject_identifier == "s"
    assert triple.relation_identifier == "r"
    assert triple.object_identifier == "o"


def test_concept_with_domain():
    concept = Concept(identifier="123", domain_identifier="d")
    assert concept.identifier == "123"
    assert concept.domain_identifier == "d"
