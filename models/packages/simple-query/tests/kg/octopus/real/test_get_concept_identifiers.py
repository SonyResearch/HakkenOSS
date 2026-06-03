import pytest

from simple_query.query.entities.inputs import (
    Argument,
    ConditionNode,
    ConditionPredicate,
    ConditionType,
)


@pytest.fixture
def condition() -> ConditionNode:
    c1 = ConditionNode(
        type=ConditionType.LEAF,
        predicate=ConditionPredicate(
            subject=Argument(value="X", is_variable=True),
            relation=Argument(value="CAUSE"),
            object=Argument(value="3893283264cd9e3c32793cf194f2efa2"),
        ),
    )
    c2 = ConditionNode(
        type=ConditionType.LEAF,
        predicate=ConditionPredicate(
            subject=Argument(value="X", is_variable=True),
            relation=Argument(value="TREAT"),
            object=Argument(value="f364fc8ce06d6a0a5c1f3867a48e8494"),
        ),
    )
    return ConditionNode(type=ConditionType.AND, children=[c1, c2])


@pytest.mark.neo4j
def test_get_concept_identifiers_with_condition(neo4j_kg, condition):
    concept_identifiers = neo4j_kg.get_concept_identifiers(
        condition=condition, domain_identifier=None
    )

    assert "0b41620452d9cf689d4784a72c1adb5a" in concept_identifiers
    assert "3b908e7369f2f92338f5622e2852476f" in concept_identifiers


@pytest.mark.neo4j
def test_get_concept_identifiers_without_condition(neo4j_kg):
    concept_identifiers = neo4j_kg.get_concept_identifiers(
        condition=None, domain_identifier="CHEMICAL"
    )

    assert len(concept_identifiers) > 0
