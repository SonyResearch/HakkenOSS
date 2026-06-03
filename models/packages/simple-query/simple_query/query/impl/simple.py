from typing import TYPE_CHECKING

from query_common.entities.query import Candidate

from simple_query.link_predictor.entities.inputs import (
    convert_predicate_to_link_predictor_input_triple,
)
from simple_query.query.base import Querying
from simple_query.query.entities.configs import SimpleQueryingConfig

if TYPE_CHECKING:
    from simple_query.query.entities.inputs import QueryInput


class SimpleQuerying(Querying[SimpleQueryingConfig]):
    def find_candidates(self, query_input: "QueryInput") -> list[Candidate]:
        candidate_concept_identifiers = self.kg.get_concept_identifiers(
            domain_identifier=query_input.variable_domain_identifier,
            condition=query_input.condition,
        )

        link_predictor_input_triples = [
            convert_predicate_to_link_predictor_input_triple(
                predicate=query_input.target_predicate,
                variable_substitution={query_input.variable_name: candidate_identifier},
            )
            for candidate_identifier in candidate_concept_identifiers
        ]
        probs = self.link_predictor.predict(link_predictor_input_triples)

        candidates = [
            Candidate(
                var_assignments={query_input.variable_name: concept_identifier},
                condition_scores={},
                query_score=score,
            )
            for concept_identifier, score in zip(candidate_concept_identifiers, probs, strict=True)
        ]
        return sorted(
            candidates,
            key=lambda candidate: candidate.query_score,  # type: ignore
            reverse=True,
        )
