from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, cast

from filtering.core.entities.candidate import (
    InputNodeCandidate as FilteringInputNodeCandidate,
)
from loguru import logger
from query_common.entities.conditions.base import AtomicCondition, Condition
from query_common.entities.conditions.link import LinkCondition
from query_common.entities.conditions.logical import NegatedCondition
from query_common.entities.kg.triple import Triple
from query_common.entities.query import Candidate
from query_common.entities.variable import Variable, VarLabel

from complex_query.core.values.errors import (
    SearchInputError,
    SearchLogicError,
)
from complex_query.impl.search.beam_search.cqd_representations import (
    QueryConditionStep,
    QueryPartialSolution,
)
from complex_query.impl.search.beam_search.cqd_utils import get_execution_order
from complex_query.impl.search.beam_search.generic.problem import ProblemSimulator

if TYPE_CHECKING:
    from collections.abc import Sequence

    from filtering.core.contracts import NodeFiltering
    from query_common.entities.kg.identifier import ConceptIdentifier

    from complex_query.core.contracts.kg import KnowledgeGraph
    from complex_query.core.contracts.link_predictor import LinkPredictor
    from complex_query.core.contracts.score_ledger import ScoreLedger


class QuerySimulator(ProblemSimulator[QueryPartialSolution, QueryConditionStep]):
    def __init__(
        self,
        kg: KnowledgeGraph,
        ledger: ScoreLedger,
        link_predictor: LinkPredictor,
        conjunctive_conditions: Sequence[Condition],
        node_filtering: NodeFiltering | None = None,
    ):
        self.kg = kg
        self.ledger = ledger
        self.link_predictor = link_predictor
        self.conjunctive_conditions = conjunctive_conditions
        self.execution_order = get_execution_order(conjunctive_conditions)
        self.node_filtering = node_filtering

    def expand_solution(
        self,
        partial_solution: QueryPartialSolution,
        step: QueryConditionStep,
        step_score: float,
    ) -> QueryPartialSolution:
        if step.condition.id in partial_solution.candidate.condition_scores:
            raise SearchLogicError(
                f"Condition {step.condition.id} already in solution, cannot add it."
            )
        partial_assignment = partial_solution.candidate.var_assignments
        extension_assignment = step.assignment
        partial_scores = partial_solution.candidate.condition_scores
        extension_scores = {step.condition.id: step_score}
        for v in partial_assignment:
            if v in extension_assignment and partial_assignment[v] != extension_assignment[v]:
                raise SearchLogicError(
                    "The step does not match the partial solution. Their variable assignments "
                    "do not match."
                )
        merged_assignment = partial_assignment | extension_assignment
        merged_scores = partial_scores | extension_scores
        return QueryPartialSolution(
            candidate=Candidate(var_assignments=merged_assignment, condition_scores=merged_scores)
        )

    def evaluate_condition(
        self, condition: Condition, assignment: dict[VarLabel, ConceptIdentifier]
    ) -> float:
        if isinstance(condition, NegatedCondition):
            return 1 - self.evaluate_condition(condition.condition, assignment)
        if isinstance(condition, LinkCondition):
            triple = self.ground_triple_using_assignment(condition, assignment)
            score = self.predict_triple_score(triple)
        else:
            raise SearchInputError(f"Condition type {type(condition)} not supported.")
        return score

    def evaluate_step(
        self,
        partial_solution: QueryPartialSolution,  # noqa: ARG002
        step: QueryConditionStep,
    ) -> float:
        return self.evaluate_condition(step.condition, step.assignment)

    def evaluate_batch_of_steps(
        self,
        partial_solution: QueryPartialSolution,
        steps: list[QueryConditionStep],
    ) -> list[float]:
        condition_ids = [step.condition.id for step in steps]
        if not all(isinstance(c_id, type(condition_ids[0])) for c_id in condition_ids):
            logger.warning(
                "All conditions are not the same type. "
                "Evaluating them one-by-one instead of by batch."
            )
            scores = [self.evaluate_step(partial_solution, step) for step in steps]
        elif all(isinstance(step.condition, NegatedCondition) for step in steps):
            negated_batch = [
                QueryConditionStep(
                    cast("NegatedCondition", step.condition).condition, step.assignment
                )
                for step in steps
            ]
            scores = [
                1 - score for score in self.evaluate_batch_of_steps(partial_solution, negated_batch)
            ]
        elif all(isinstance(step.condition, LinkCondition) for step in steps):
            triples = [
                self.ground_triple_using_assignment(
                    cast("LinkCondition", step.condition), step.assignment
                )
                for step in steps
            ]
            scores = self.predict_batch_triple_score(triples)
        else:
            logger.info(
                f"Batch evaluation not implemented for {type(steps[0].condition)}. "
                "Evaluating them one-by-one instead of by batch."
            )
            scores = [self.evaluate_step(partial_solution, step) for step in steps]
        return scores

    def possible_steps_from_solution(
        self,
        partial_solution: QueryPartialSolution,
        ignore_search_space_exception: bool = False,
    ) -> list[QueryConditionStep]:
        """Find the next condition and assignments to run."""
        condition = self.get_next_condition_to_run(partial_solution)
        steps = []
        logger.info(f"Next condition is {condition!s}")
        if isinstance(condition, LinkCondition | NegatedCondition):
            if isinstance(condition, NegatedCondition) and not isinstance(
                condition.condition, AtomicCondition
            ):
                raise SearchLogicError("Expected a negation only on an atomic condition.")
            if (
                sum(
                    [
                        not partial_solution.is_variable_already_assigned(v.label)
                        for v in condition.variables()
                    ]
                )
                > 1
            ):
                if ignore_search_space_exception:
                    logger.warning(
                        f"Found two non-assigned variables in {condition}. "
                        "Proceeding anyway but the search space will be large!"
                    )
                else:
                    raise SearchInputError(
                        f"Found two non-assigned variables in {condition}. Aborting query due to "
                        "large search space. Make sure you are submitting a query "
                        "correctly anchored."
                    )
            variables_to_assignments = {
                var_.label: self.possible_assignments_for_variable_from_solution(
                    var_,
                    partial_solution,
                    negated=isinstance(condition, NegatedCondition),
                )
                for var_ in condition.variables()
            }
            for assignment_instance in itertools.product(*list(variables_to_assignments.values())):
                assignment = dict(
                    zip(variables_to_assignments.keys(), assignment_instance, strict=False)
                )
                # TODO: When working with a Link Condition, Should we filter out every assignment
                #  that already exists in the Graph?? And not propose such groundtruth candidates?
                steps.append(QueryConditionStep(condition=condition, assignment=assignment))
        else:
            logger.exception(f"Condition type {type(condition)} not supported.")
            raise SearchInputError(f"Condition type {type(condition)} not supported.")
        logger.info(f"Found {len(steps)} steps to new candidate solutions.")
        return steps

    def possible_assignments_for_variable_from_solution(
        self,
        variable: Variable,
        solution: QueryPartialSolution,
        negated: bool = False,  # noqa: ARG002
    ) -> list[ConceptIdentifier]:
        if solution.is_variable_already_assigned(variable.label):
            concept_ids = [solution.get_assignment_for_variable(variable.label)]
        else:
            # TODO: This is where we would add the filtering module
            # The negated keyword argument is because the output of the filtering may be different
            # if the condition is negated or not (?)
            concept_ids = [
                s.identifier for s in self.kg.get_concepts_from_domain(variable.domain_identifier)
            ]
            if self.node_filtering:
                filtering_input_candidates = [
                    FilteringInputNodeCandidate(node_id=concept_id) for concept_id in concept_ids
                ]
                concept_ids = [
                    output_candidate.node_id
                    for output_candidate in self.node_filtering.filter(
                        candidates=filtering_input_candidates
                    )
                ]
        return concept_ids

    def is_solution_complete(self, partial_solution: QueryPartialSolution) -> bool:
        condition_scores = partial_solution.candidate.condition_scores
        all_conditions_answered = all(
            condition.id in condition_scores for condition in self.conjunctive_conditions
        )
        if (
            len(condition_scores) == len(self.conjunctive_conditions)
            and not all_conditions_answered
        ):
            conditions_missing_scores = [
                condition
                for condition in self.conjunctive_conditions
                if condition.id not in condition_scores
            ]
            unknown_cids_found_in_scores = [
                id_
                for id_ in condition_scores
                if id_ not in [c.id for c in self.conjunctive_conditions]
            ]
            raise SearchLogicError(
                "Found as many scores as there was conditions, but conditions IDs"
                " do not match exactly. Conditions that miss scores: "
                f"{[str(c) + ' with ID ' + str(c.id) for c in conditions_missing_scores]}. Unknown"
                f" conditions IDs found in scores: {unknown_cids_found_in_scores}."
            )
        if all_conditions_answered:
            all_variables = [
                v.label for condition in self.conjunctive_conditions for v in condition.variables()
            ]
            all_variables = list(set(all_variables))
            if not all(v in partial_solution.candidate.var_assignments for v in all_variables):
                raise SearchLogicError(
                    "All conditions have been assigned scores but some "
                    "variables have not been assigned."
                )
        return all_conditions_answered

    def get_next_condition_to_run(self, partial_solution: QueryPartialSolution) -> Condition:
        step_idx = len(partial_solution.candidate.condition_scores)
        if step_idx >= len(self.conjunctive_conditions):
            raise SearchLogicError("No more conditions to run.")
        return self.conjunctive_conditions[self.execution_order[step_idx]]

    @staticmethod
    def ground_triple_using_assignment(
        condition: LinkCondition, assignment: dict[str, ConceptIdentifier]
    ):
        subject_identifier = (
            assignment[condition.subject.label]
            if isinstance(condition.subject, Variable)
            else condition.subject.identifier
        )
        object_identifier = (
            assignment[condition.object.label]
            if isinstance(condition.object, Variable)
            else condition.object.identifier
        )

        return Triple(
            subject_identifier=subject_identifier,
            relation_identifier=condition.relation.identifier,
            object_identifier=object_identifier,
        )

    def predict_triple_score(self, triple: Triple) -> float:
        try:
            score = self.ledger.retrieve_link_score(triple)
        except KeyError:
            score = self.link_predictor.predict(triple)
            self.ledger.save_link_score(triple, score)
        return score

    def predict_batch_triple_score(self, triples: list[Triple]) -> list[float]:
        # Separate triples to predict from the ones available in the ledger
        final_scores, triples_to_predict, original_indices = [], [], []
        for i, triple in enumerate(triples):
            try:
                score = self.ledger.retrieve_link_score(triple)
                final_scores.append((i, score))
            except KeyError:
                triples_to_predict.append(triple)
                original_indices.append(i)
        # Predict scores for triples not in ledger
        if triples_to_predict:
            predicted_scores = self.link_predictor.predict_batch(triples_to_predict)
            for score, triple in zip(predicted_scores, triples_to_predict, strict=False):
                self.ledger.save_link_score(triple, score)
            # Add predicted scores to final_scores list
            for orig_idx, score in zip(original_indices, predicted_scores, strict=False):
                final_scores.append((orig_idx, score))
        # Sort the final_scores list by the original index and extract only the scores
        final_scores.sort(key=lambda x: x[0])

        return [score for _, score in final_scores]
