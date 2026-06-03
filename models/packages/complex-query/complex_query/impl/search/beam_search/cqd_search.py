from __future__ import annotations

import itertools
from collections import defaultdict
from typing import TYPE_CHECKING

from loguru import logger
from query_common.entities.conditions.link import LinkCondition
from query_common.entities.conditions.logical import (
    ConjunctiveCondition,
    DisjunctiveCondition,
    NegatedCondition,
)
from query_common.entities.query import Candidate

from complex_query.core.contracts.search import Search
from complex_query.core.entities.config.search import BeamSearchConfig
from complex_query.core.values.errors import SearchInputError
from complex_query.impl.search.beam_search.cqd_representations import (
    QueryConditionStep,
    QueryPartialSolution,
)
from complex_query.impl.search.beam_search.cqd_simulator import QuerySimulator
from complex_query.impl.search.beam_search.generic.beam import BeamSearch

if TYPE_CHECKING:
    from collections.abc import Sequence

    from query_common.entities.conditions.base import Condition
    from query_common.entities.grounded_query import GroundedQuery
    from query_common.entities.kg.identifier import ConceptIdentifier
    from query_common.entities.variable import VarLabel


class QueryBeamSearch(Search[BeamSearchConfig]):
    def find_candidates(
        self,
        query: GroundedQuery,
        n_candidates: int,
    ) -> list[Candidate]:
        batched = self.config.batched
        batch_size = self.config.batch_size
        beam_size = self.config.beam_size

        if beam_size < n_candidates:
            logger.warning(
                f"Beam size ({beam_size}) must be greater than or equal to the"
                f" number of answer candidates ({n_candidates}) for optimal results."
            )

        if isinstance(query.condition, DisjunctiveCondition):
            disjunctive_conditions = query.condition.flattened_conditions()
        else:
            disjunctive_conditions = [query.condition]

        # Runs a beam search for N CNF subquery, collecting the top K candidates for each part.
        # Stores results as a list of N lists, where each sublist contains K ranked candidates.
        top_k_per_part: list[list[Candidate]] = []
        all_conjunctive_subconditions: list[list[Condition]] = []

        for condition in disjunctive_conditions:
            if isinstance(condition, ConjunctiveCondition):
                conjunctive_subconditions = condition.flattened_conditions()
            else:
                conjunctive_subconditions = [condition]

            for subcondition in conjunctive_subconditions:
                if not isinstance(subcondition, LinkCondition | NegatedCondition):
                    raise SearchInputError(
                        f"The query should be in a DNF format, "
                        f"however non-literal condition is found in "
                        f"{subcondition!s}, which is a part of {query.condition!s}."
                    )

            all_conjunctive_subconditions.append(conjunctive_subconditions)
            query_simulator = QuerySimulator(
                kg=self.kg,
                ledger=self.ledger,
                link_predictor=self.link_predictor,
                conjunctive_conditions=conjunctive_subconditions,
                node_filtering=self.node_filtering,
            )
            beam_search = BeamSearch[QuerySimulator, QueryPartialSolution, QueryConditionStep](
                problem_simulator=query_simulator,
                beam_size=beam_size,
                stop_at_first_final_solution=False,
                score_aggregator=self.score_aggregator,
                batch_steps=batched,
                batch_size=batch_size,
            )
            initial_solution = QueryPartialSolution.from_empty()
            beam_nodes = beam_search.search(initial_solution)
            beam_nodes = sorted(beam_nodes, key=lambda x: x.cumulative_score, reverse=True)
            beam_nodes = beam_nodes[:n_candidates]

            candidates = []
            for beam_node in beam_nodes:
                candidate = beam_node.partial_solution.candidate
                candidate.query_score = beam_node.cumulative_score
                candidates.append(candidate)

            top_k_per_part.append(candidates)

        if len(top_k_per_part) == 1:
            combined_candidates = top_k_per_part[0]
        else:
            combined_candidates = self.combine_candidates(
                top_k_per_part, all_conjunctive_subconditions
            )

        return combined_candidates

    def combine_candidates(
        self,
        top_k_per_part: Sequence[Sequence[Candidate]],
        all_conjunctive_subconditions: list[list[Condition]],
    ) -> list[Candidate]:
        possible_assignments_by_var: dict[VarLabel, set[ConceptIdentifier]] = defaultdict(set)
        for candidates in top_k_per_part:
            for candidate in candidates:
                for var_label in candidate.var_assignments:
                    possible_assignments_by_var[var_label].add(candidate.var_assignments[var_label])

        # Create the cartesian product of all assignment sets
        all_vars = list(possible_assignments_by_var.keys())
        all_combinations = itertools.product(*(possible_assignments_by_var[v] for v in all_vars))
        all_assignments: list[dict[VarLabel, ConceptIdentifier]] = [
            dict(zip(all_vars, combo, strict=False)) for combo in all_combinations
        ]

        # Re-run the simulator for the assignments in the cartesian product.
        # Probably not the most efficient way to do it, as some calculations are repeated,
        # but (i) there shouldn't be that many assignments anyway (ii) the score ledger should help.
        simulator = QuerySimulator(
            kg=self.kg,
            ledger=self.ledger,
            link_predictor=self.link_predictor,
            conjunctive_conditions=[],
            node_filtering=self.node_filtering,
        )
        all_candidates: list[Candidate] = []
        for assignment in all_assignments:
            cnf_scores: list[float] = []
            all_condition_id_to_score = {}
            for conjunctive_subconditions in all_conjunctive_subconditions:
                condition_id_to_score = {
                    condition.id: simulator.evaluate_condition(condition, assignment)
                    for condition in conjunctive_subconditions
                }
                cnf_scores.append(
                    self.score_aggregator.t_norm(list(condition_id_to_score.values()))
                )
                all_condition_id_to_score.update(condition_id_to_score)
            total_score = self.score_aggregator.t_conorm(cnf_scores)
            all_candidates.append(
                Candidate(
                    var_assignments=assignment,
                    condition_scores=all_condition_id_to_score,
                    query_score=total_score,
                )
            )

        return all_candidates
