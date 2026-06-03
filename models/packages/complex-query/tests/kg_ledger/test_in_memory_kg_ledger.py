import pytest
from query_common.entities.kg.concept import Concept

from complex_query.core.entities.config.kg_ledger import InMemoryKGLedgerConfig
from complex_query.impl.kg_ledger.in_memory import InMemoryKnowledgeGraphLedger


@pytest.fixture
def ledger():
    config = InMemoryKGLedgerConfig()
    return InMemoryKnowledgeGraphLedger(config)


def test_add_concept(ledger):
    concept = Concept(identifier="id1", label="Test Concept", domain_identifier="domain1")
    ledger.add_concept(concept)
    assert "domain1" not in ledger._domain_concepts_cache

    retrieved_concept = ledger.get_concept("id1")
    assert retrieved_concept.identifier == concept.identifier
    assert retrieved_concept.label == concept.label
    assert retrieved_concept.domain_identifier == concept.domain_identifier

    with pytest.raises(KeyError):
        ledger.get_concepts_from_domain("domain1")


def test_add_concepts_for_domain(ledger):
    concepts = [
        Concept(identifier=str(i), label=f"Test Concept {i}", domain_identifier="d")
        for i in range(3)
    ]
    ledger.add_concepts_for_domain(concepts, "d")

    assert "d" in ledger._domain_concepts_cache
    assert len(ledger._concept_cache) == 3
    retrieved_concepts = ledger.get_concepts_from_domain("d")
    assert len(retrieved_concepts) == 3


def test_get_concepts_from_domain(ledger):
    domain_1 = "d1"
    domain_2 = "d2"
    concepts = [
        Concept(identifier=str(i), label=f"Test Concept {i}", domain_identifier=domain_1)
        for i in range(3)
    ]

    ledger.add_concepts_for_domain(concepts, domain_1)
    for concept in concepts:
        ledger.add_concept(concept)

    ledger.add_concept(Concept(identifier="3", label="Test Concept 3", domain_identifier=domain_2))

    domain_1_concepts = ledger.get_concepts_from_domain(domain_1)
    assert len(domain_1_concepts) == 3
    assert all(concept.domain_identifier == domain_1 for concept in domain_1_concepts)

    with pytest.raises(KeyError):
        # Should raise KeyError, if not added by `add_concepts_for_domain`
        ledger.get_concepts_from_domain(domain_2)


def test_get_concepts_from_nonexistent_domain(ledger):
    with pytest.raises(KeyError):
        ledger.get_concepts_from_domain("_NOT_EXIST_DOMAIN_")


def test_get_concept(ledger):
    concept = Concept(identifier="1", label="Test Concept", domain_identifier="d1")
    ledger.add_concept(concept)

    retrieved_concept = ledger.get_concept("1")
    assert retrieved_concept.identifier == concept.identifier
    assert retrieved_concept.label == concept.label
    assert retrieved_concept.domain_identifier == concept.domain_identifier


def test_get_nonexistent_concept(ledger):
    with pytest.raises(KeyError):
        ledger.get_concept("1")


def test_add_multiple_concepts_same_domain(ledger):
    concepts = [
        Concept(identifier=str(i), label=f"Test Concept {i}", domain_identifier="d1")
        for i in range(5)
    ]
    ledger.add_concepts_for_domain(concepts, domain_identifier="d1")

    retrieved_concepts = ledger.get_concepts_from_domain("d1")
    assert len(retrieved_concepts) == 5
    assert all(c.domain_identifier == "d1" for c in retrieved_concepts)


def test_add_concepts_multiple_domains(ledger):
    domains = ["d1", "d2", "d3"]
    for d in domains:
        ledger.add_concepts_for_domain(
            [
                Concept(identifier=f"{d}_{i}", label=f"Test Concept {d}_{i}", domain_identifier=d)
                for i in range(3)
            ],
            domain_identifier=d,
        )

    for d in domains:
        retrieved_concepts = ledger.get_concepts_from_domain(d)
        assert len(retrieved_concepts) == 3
        assert all(c.domain_identifier == d for c in retrieved_concepts)


def test_add_concepts_mixed_domains(ledger):
    with pytest.raises(ValueError):
        ledger.add_concepts_for_domain(
            [
                Concept(identifier=f"{i}", label=f"Test Concept {i}", domain_identifier=f"d_{i}")
                for i in range(3)
            ],
            domain_identifier="d",
        )
