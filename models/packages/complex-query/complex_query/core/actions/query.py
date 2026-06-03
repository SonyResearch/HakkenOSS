from typing import TYPE_CHECKING, Any

from dependency_injector.wiring import Provide, inject
from loguru import logger
from query_common.entities.query import QueryRequest, QueryResponse
from query_common.grounding.actions import ground_query_given_variables

from complex_query.core.actions.parse import parse_query
from complex_query.core.actions.search import search_candidates

if TYPE_CHECKING:
    from query_common.parse.base import Parser

    from complex_query.core.contracts import Search


@inject
def answer_query(
    request: "QueryRequest",
    parser: "Parser" = Provide["parser"],
    search: "Search" = Provide["search"],
) -> Any:
    query_clause = parse_query(query_string=request.formula, parser=parser)
    grounded_query = ground_query_given_variables(
        query_clause, variables=[v.to_query_variable() for v in request.variables]
    )
    logger.info(f"Received query: {grounded_query}")
    candidates = search_candidates(
        search_method=search,
        query=grounded_query,
        n_candidates=request.n_candidates,
    )
    # TODO: Add condition information in the response
    return QueryResponse(candidates=candidates)
