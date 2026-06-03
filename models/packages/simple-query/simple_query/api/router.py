from typing import TYPE_CHECKING

from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends
from loguru import logger
from query_common.entities.query import Candidate, QueryRequest, QueryResponse

from simple_query.api.entities import ConstraintFilteringRequest
from simple_query.api.errors import QueryInputError
from simple_query.kg.base import KnowledgeGraph
from simple_query.kg.entities.constraint import ConstraintFilteringOutput
from simple_query.query.entities.inputs import convert_to_query_input

if TYPE_CHECKING:
    from query_common.parse.base import Parser

    from simple_query.query.base import Querying

router = APIRouter()


@inject
def find_candidates_from_query_request(
    request: QueryRequest,
    parser: "Parser" = Provide["parser"],
    querying: "Querying" = Provide["querying"],
) -> list["Candidate"]:
    if len(request.variables) != 1:
        raise QueryInputError(
            f"request should have 1 variable at most, but got {request.variables}"
        )

    variable_name = request.variables[0].label
    variable_domain = request.variables[0].domain

    parsed_query = parser.parse_query(request.formula)

    simple_query_input = convert_to_query_input(
        query=parsed_query, variable_name=variable_name, variable_domain_identifier=variable_domain
    )

    candidates = querying.find_candidates(simple_query_input)
    return candidates[: request.n_candidates]


@router.post("/query", response_model=QueryResponse)
def query(
    request: QueryRequest,
):
    logger.info(f"Received query request with formula: {request.formula}")
    candidates = find_candidates_from_query_request(request)
    return QueryResponse(candidates=candidates)


@router.post("/filter_constraint", response_model=ConstraintFilteringOutput)
@inject
def filter_constraint(
    request: ConstraintFilteringRequest,
    kg: KnowledgeGraph = Depends(Provide["kg"]),  # noqa: B008
):
    logger.info(f"Received constraint filtering request: {request}")
    return kg.filter_constraint(request)
