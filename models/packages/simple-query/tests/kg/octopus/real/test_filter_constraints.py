import pytest
from query_common.entities.kg.concept import Concept
from query_common.entities.kg.relation import Relation

from simple_query.kg.entities.constraint import TripleConstraint, TripleConstraintArgument
from simple_query.kg.values.errors import Neo4jKGError


@pytest.fixture
def triple_constraint() -> TripleConstraint:
    return TripleConstraint(
        subject=TripleConstraintArgument(value="0003fb22e2049da4ab2c993efa06f726"),
        relation=TripleConstraintArgument(value="R", is_variable=True),
        object=TripleConstraintArgument(value="X", is_variable=True),
    )


@pytest.fixture
def triple_constraint_with_domain() -> TripleConstraint:
    return TripleConstraint(
        subject=TripleConstraintArgument(value="0003fb22e2049da4ab2c993efa06f726"),
        relation=TripleConstraintArgument(value="R", is_variable=True),
        object=TripleConstraintArgument(value="X", is_variable=True, domain_identifier="GENE"),
    )


@pytest.fixture
def erroneous_triple_constraint() -> TripleConstraint:
    return TripleConstraint(
        subject=TripleConstraintArgument(value="0003fb22e2049da4ab2c993efa06f726"),
        relation=TripleConstraintArgument(value="R", is_variable=True, domain_identifier="GENE"),
        object=TripleConstraintArgument(value="X", is_variable=True, domain_identifier="CHEMICAL"),
    )


@pytest.fixture
def triple_constraint_without_condition() -> TripleConstraint:
    return TripleConstraint(
        subject=TripleConstraintArgument(value="X", is_variable=True),
        relation=TripleConstraintArgument(value="R", is_variable=True),
        object=TripleConstraintArgument(value="Y", is_variable=True),
    )


@pytest.mark.neo4j
def test_filter_constraint(neo4j_kg, triple_constraint):
    constraint_filtering_output = neo4j_kg.filter_constraint(triple_constraint=triple_constraint)
    for entry in constraint_filtering_output:
        if entry.variable == "R":
            assert entry.type == "relation"
            assert len(entry.values) > 0
            assert all(isinstance(v, Relation) for v in entry.values)
        if entry.variable == "X":
            assert entry.type == "concept"
            assert len(entry.values) > 0
            assert all(isinstance(v, Concept) for v in entry.values)


@pytest.mark.neo4j
def test_filter_constraint_with_domain(neo4j_kg, triple_constraint_with_domain):
    constraint_filtering_output = neo4j_kg.filter_constraint(
        triple_constraint=triple_constraint_with_domain
    )
    for entry in constraint_filtering_output:
        if entry.variable == "R":
            assert entry.type == "relation"
            assert len(entry.values) > 0
            assert all(isinstance(v, Relation) for v in entry.values)
        if entry.variable == "X":
            assert entry.type == "concept"
            assert len(entry.values) > 0
            assert all(isinstance(v, Concept) for v in entry.values)
            assert all(v.domain_identifier == "GENE" for v in entry.values)


@pytest.mark.neo4j
def test_filter_constraint_errors(
    neo4j_kg, erroneous_triple_constraint, triple_constraint_without_condition
):
    with pytest.raises(Neo4jKGError):
        neo4j_kg.filter_constraint(erroneous_triple_constraint)
    with pytest.raises(Neo4jKGError):
        neo4j_kg.filter_constraint(triple_constraint_without_condition)
