import functools
import operator
from typing import TYPE_CHECKING, Any, ClassVar

from loguru import logger
from neo4j import GraphDatabase
from query_common.entities.kg.concept import Concept
from query_common.entities.kg.relation import Relation

from simple_query.kg.base import KnowledgeGraph
from simple_query.kg.entities.configs import Neo4jKnowledgeGraphConfig
from simple_query.kg.entities.constraint import (
    ConstraintFilteringOutputEntry,
    DomainIdentifier,
)
from simple_query.kg.values.errors import Neo4jKGError
from simple_query.query.values.types import ConditionType

if TYPE_CHECKING:
    from query_common.entities.kg.identifier import ConceptIdentifier

    from simple_query.kg.entities.constraint import (
        ConstraintFilteringOutput,
        TripleConstraint,
    )
    from simple_query.query.entities.inputs import ConditionNode, ConditionPredicate


class Neo4jKnowledgeGraph(KnowledgeGraph[Neo4jKnowledgeGraphConfig]):
    # Variable names used in Neo4j query
    SUBJECT_VARIABLE: ClassVar[str] = "s"
    RELATION_VARIABLE: ClassVar[str] = "r"
    OBJECT_VARIABLE: ClassVar[str] = "o"

    def __init__(self, config: Neo4jKnowledgeGraphConfig) -> None:
        super().__init__(config)

        self.driver = GraphDatabase.driver(
            self.config.base_url, auth=(self.config.username, self.config.password)
        )

    def _execute_neo4j_query(self, neo4j_query: str) -> list[dict[str, Any]]:
        logger.info(f"Executing query to Neo4j: {neo4j_query}")

        with self.driver.session() as session:
            query_result = session.run(neo4j_query)

            results = query_result.data()
            summary = query_result.consume()
            logger.info(f"Query took {summary.result_available_after}ms")

        return results

    def _get_neo4j_variable_name_from_condition_predicate(
        self, condition_predicate: "ConditionPredicate"
    ) -> str:
        variable_name: str | None = None

        if condition_predicate.subject.is_variable:
            variable_name = self.SUBJECT_VARIABLE
        if condition_predicate.relation.is_variable:
            raise Neo4jKGError(f"relation could not be a variable: {condition_predicate}")
        if condition_predicate.object.is_variable:
            if variable_name is not None:
                raise Neo4jKGError(f"predicate has multiple variables: {condition_predicate}")
            variable_name = self.OBJECT_VARIABLE

        if variable_name is None:
            raise Neo4jKGError(f"could not find a variable: {condition_predicate}")

        return variable_name

    def _convert_condition_predicate_to_neo4j_condition(
        self, condition_predicate: "ConditionPredicate"
    ) -> str:
        subconditions = []

        if not condition_predicate.subject.is_variable:
            subconditions.append(
                f"{self.SUBJECT_VARIABLE}.node_id = '{condition_predicate.subject.value}'"
            )

        if not condition_predicate.relation.is_variable:
            subconditions.append(
                f"TYPE({self.RELATION_VARIABLE}) = '{condition_predicate.relation.value}'"
            )

        if not condition_predicate.object.is_variable:
            subconditions.append(
                f"{self.OBJECT_VARIABLE}.node_id = '{condition_predicate.object.value}'"
            )

        return " AND ".join(subconditions)

    def _convert_condition_predicate_to_condition_before_exists(
        self, condition_predicate: "ConditionPredicate"
    ) -> str:
        if not condition_predicate.subject.is_variable:
            return f"{self.SUBJECT_VARIABLE}.node_id = '{condition_predicate.subject.value}'"

        if not condition_predicate.object.is_variable:
            return f"{self.OBJECT_VARIABLE}.node_id = '{condition_predicate.object.value}'"

        return ""

    def _get_relation_value(self, condition_predicate: "ConditionPredicate") -> str | None:
        if not condition_predicate.relation.is_variable:
            return condition_predicate.relation.value

        return None

    def _get_concept_identifier_set_for_condition(
        self, condition: "ConditionNode", domain_identifier: DomainIdentifier | None
    ) -> set["ConceptIdentifier"]:
        if condition.type in (ConditionType.AND, ConditionType.OR):
            children_identifier_sets = [
                self._get_concept_identifier_set_for_condition(
                    condition=child, domain_identifier=domain_identifier
                )
                for child in condition.children
            ]
            reduce_operator = operator.and_ if condition.type == ConditionType.AND else operator.or_

            return functools.reduce(reduce_operator, children_identifier_sets)

        # NOT or LEAF
        if condition.type == ConditionType.LEAF:
            predicate = condition.predicate
        else:
            if len(condition.children) != 1:
                raise Neo4jKGError("NOT condition node should only have one child")
            predicate = condition.children[0].predicate

        if predicate is None:
            raise Neo4jKGError("predicate of a LEAF condition node cannot be empty")

        variable_name = self._get_neo4j_variable_name_from_condition_predicate(predicate)

        neo4j_query_condition = self._convert_condition_predicate_to_neo4j_condition(predicate)
        condition_before_exists = self._convert_condition_predicate_to_condition_before_exists(
            predicate
        )
        domain_identifier_condition = (
            f"'{domain_identifier}' IN LABELS({variable_name})" if domain_identifier else ""
        )
        relation_value = self._get_relation_value(predicate)

        not_condition = None
        if condition.type == ConditionType.NOT:
            if domain_identifier:
                not_condition = (
                    f"NOT '{relation_value}' IN COLLECT "
                    "{ "
                    f"MATCH "
                    f"({self.SUBJECT_VARIABLE})"
                    f"-[r2]"
                    f"->({self.OBJECT_VARIABLE}) "
                    f"WHERE {condition_before_exists} AND {domain_identifier_condition} "
                    "RETURN TYPE(r2) "
                    "}"
                )
            else:
                not_condition = (
                    f"NOT '{relation_value}' IN COLLECT "
                    "{ "
                    f"MATCH "
                    f"({self.SUBJECT_VARIABLE})"
                    f"-[r2]"
                    f"->({self.OBJECT_VARIABLE}) "
                    f"WHERE {condition_before_exists} "
                    "RETURN TYPE(r2) "
                    "}"
                )

        final_condition = (
            f"{condition_before_exists}" if not_condition else f"{neo4j_query_condition}"
        )
        if domain_identifier:
            final_condition = f"{final_condition} AND {domain_identifier_condition}"

        if not_condition:
            final_condition = f"{final_condition} AND {not_condition}"

        neo4j_query = (
            "MATCH (s)-[r]->(o) "
            f"WHERE {final_condition} "
            f"RETURN DISTINCT({variable_name}.node_id) AS node_id"
        )
        rows = self._execute_neo4j_query(neo4j_query)
        return {row["node_id"] for row in rows}

    def get_concept_identifiers(
        self, domain_identifier: DomainIdentifier | None, condition: "ConditionNode | None"
    ) -> list["ConceptIdentifier"]:
        if condition is None:
            # If condition is None, directly query Neo4j to obtain concepts
            if domain_identifier is None:
                raise Neo4jKGError(
                    "querying without domain nor condition will occur a heavy load "
                    "to the Neo4j server."
                )
            neo4j_query = f"MATCH (n:{domain_identifier}) RETURN DISTINCT(n.node_id) AS node_id"
            rows = self._execute_neo4j_query(neo4j_query)

            return [row["node_id"] for row in rows]

        return list(
            self._get_concept_identifier_set_for_condition(
                condition=condition, domain_identifier=domain_identifier
            )
        )

    def filter_constraint(
        self, triple_constraint: "TripleConstraint"
    ) -> "ConstraintFilteringOutput":
        neo4j_conditions: list[str] = []
        neo4j_return_statements: list[str] = []

        concept_variable_set = set()
        relation_variable_set = set()

        for concept_argument, neo4j_variable in zip(
            [triple_constraint.subject, triple_constraint.object],
            [self.SUBJECT_VARIABLE, self.OBJECT_VARIABLE],
            strict=True,
        ):
            if concept_argument.is_variable:
                neo4j_return_statements.append(
                    f"COLLECT(DISTINCT {{ "
                    f"  node_name: {neo4j_variable}.node_name, "
                    f"  node_id: {neo4j_variable}.node_id, "
                    f"  domain_identifier: LABELS({neo4j_variable})[0] "
                    f"}}) AS {concept_argument.value}"
                )
                concept_variable_set.add(concept_argument.value)
            else:
                neo4j_conditions.append(f"{neo4j_variable}.node_id = '{concept_argument.value}'")
            if concept_argument.domain_identifier is not None:
                neo4j_conditions.append(
                    f"'{concept_argument.domain_identifier}' IN LABELS({neo4j_variable})"
                )

        relation_argument = triple_constraint.relation
        relation_neo4j_variable = self.RELATION_VARIABLE
        if relation_argument.is_variable:
            neo4j_return_statements.append(
                f"COLLECT(DISTINCT {{ "
                f"  relation_type: TYPE({relation_neo4j_variable}) "
                f"}}) AS {relation_argument.value}"
            )
            relation_variable_set.add(relation_argument.value)
        else:
            neo4j_conditions.append(
                f"TYPE({relation_neo4j_variable}) = '{relation_argument.value}'"
            )
        if relation_argument.domain_identifier is not None:
            raise Neo4jKGError("relation argument could not have a non-null `domain` value")

        if not neo4j_conditions:
            raise Neo4jKGError("no condition is given in the constraint filtering query")

        neo4j_query = (
            f"MATCH ({self.SUBJECT_VARIABLE})-[{self.RELATION_VARIABLE}]->({self.OBJECT_VARIABLE}) "
            f"WHERE {' AND '.join(neo4j_conditions)} "
            f"RETURN {', '.join(neo4j_return_statements)}"
        )

        rows = self._execute_neo4j_query(neo4j_query)
        neo4j_result: dict[str, list[dict[str, str]]] = rows[0]

        output: ConstraintFilteringOutput = []
        for variable, constrained_dicts in neo4j_result.items():
            if variable in concept_variable_set:
                values = [
                    Concept(
                        identifier=item["node_id"],
                        label=item["node_name"],
                        domain_identifier=item["domain_identifier"],
                    )
                    for item in constrained_dicts
                ]
                output.append(
                    ConstraintFilteringOutputEntry(variable=variable, type="concept", values=values)
                )
            else:
                values = [
                    Relation(
                        identifier=item["relation_type"],
                        label=item["relation_type"],
                    )
                    for item in constrained_dicts
                ]
                output.append(
                    ConstraintFilteringOutputEntry(
                        variable=variable, type="relation", values=values
                    )
                )
        return output
