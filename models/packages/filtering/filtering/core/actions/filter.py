from typing import TYPE_CHECKING

from dependency_injector.wiring import Provide, inject

if TYPE_CHECKING:
    from filtering.core.contracts import NodeFiltering, TripleFiltering
    from filtering.core.entities.candidate import (
        InputNodeCandidate,
        InputTripleCandidate,
        OutputNodeCandidate,
        OutputTripleCandidate,
    )


@inject
def filter_nodes(
    candidates: list["InputNodeCandidate"],
    max_output_candidates: int | None = None,
    node_filtering: "NodeFiltering" = Provide["node_filtering_model"],
) -> list["OutputNodeCandidate"]:
    return node_filtering.filter(candidates=candidates, max_output_candidates=max_output_candidates)


@inject
def filter_triples(
    candidates: list["InputTripleCandidate"],
    max_output_candidates: int | None = None,
    triple_filtering: "TripleFiltering" = Provide["triple_filtering_model"],
) -> list["OutputTripleCandidate"]:
    return triple_filtering.filter(
        candidates=candidates, max_output_candidates=max_output_candidates
    )
