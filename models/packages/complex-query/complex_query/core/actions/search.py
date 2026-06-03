from typing import TYPE_CHECKING, Any

from dependency_injector.wiring import Provide, inject

if TYPE_CHECKING:
    from query_common.entities.grounded_query import GroundedQuery
    from query_common.entities.query import Candidate

    from complex_query.core.contracts.search import Search


@inject
def search_candidates(
    query: "GroundedQuery",
    n_candidates: int,
    search_method: "Search" = Provide["search"],
    search_parameters: dict[str, Any] | None = None,
) -> list["Candidate"]:
    if search_parameters is None:
        search_parameters = {}
    return search_method.find_candidates(query, n_candidates, **search_parameters)
