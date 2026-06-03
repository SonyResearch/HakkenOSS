from __future__ import annotations

import json
import re
import uuid
from typing import TYPE_CHECKING

from fastapi import Request, Response, status
from loguru import logger

from hakken_api_gateway.core.entities.query import (
    QueryDBModel,
    QueryId,
    QueryReqFromUpstream,
    QueryReqToDownstream,
    QueryRespFromDownstream,
    QueryRespToUpstream,
    SearchParameters,
)
from hakken_api_gateway.handlers.common import forward_request

if TYPE_CHECKING:
    from hakken_api_gateway.impl.database.postgres import PostgresDatabase


def parse_formula(formula: str) -> str:
    # Regex pattern to match P(...) or EXISTS(...) with 3 arguments
    pattern = r"\b(P|EXISTS)\(\s*([^,]+),\s*([^,]+),\s*([^)]+)\)"

    def replacement_logic(match):
        # Extract the function name (Group 1)
        fn = match.group(1)

        # Extract and trim the 3 arguments (Groups 2, 3, and 4)
        # We strip whitespace from the captured groups immediately
        args = [match.group(i).strip() for i in range(2, 5)]
        formatted_args = [arg if arg == "X" else f"'{arg}'" for arg in args]

        return f"{fn}({', '.join(formatted_args)})"

    return re.sub(pattern, replacement_logic, formula)


def delete_all_user_queries(
    user_id: str,
    db: PostgresDatabase,
) -> bool:
    """Delete all user queries from the database."""

    deleted = db.delete_all_user_queries(user_id, "query")

    if deleted:
        logger.info(f"All queries for user ID {user_id} deleted from Postgres database.")
        return True
    logger.info(f"No queries for user ID {user_id} found in Postgres database.")
    return False


def delete_query(
    query_id: str,
    user_id: str,
    db: PostgresDatabase,
) -> bool:
    """Delete user query from the database."""

    logger.info(f"Deleting query with ID {query_id}")
    deleted = db.delete_query(query_id, user_id, "query")

    if deleted:
        logger.info(f"Query with ID {query_id} deleted from Postgres database.")
        return True
    logger.info(f"No query with ID {query_id} found in Postgres database.")
    return False


def get_user_queries(
    user_data: dict,
    db: PostgresDatabase,
) -> list | None:
    """Get user query from the database."""
    # Placeholder function to demonstrate where to insert logic
    logger.info("Getting user queries...")
    user_id = user_data["user_id"]

    query = db.get_user_queries(user_id, "query")

    if query:
        logger.info(f"Queries with user ID {user_id} found in Postgres database.")
        return query
    logger.info(f"No query with user ID {user_id} found in Postgres database.")
    return None


def get_query(
    query_id: QueryId,
    db: PostgresDatabase,
) -> dict | None:
    """Get user query from the database."""
    # Placeholder function to demonstrate where to insert logic
    logger.info(f"Getting user query with ID: {query_id.id}")

    query = db.get_query(query_id.id, "query")

    if query:
        logger.info(f"Query with ID {query_id} found in Postgres database.")
        return query
    logger.info(f"No query with ID {query_id} found in Postgres database.")
    return None


def insert_user_query(
    user_data: dict,
    query_body: QueryReqFromUpstream,
    query_response: QueryRespFromDownstream,
    db: PostgresDatabase,
) -> str | None:
    """Insert user query and response into the database."""

    logger.info("Inserting user query ...")

    user_id = user_data.get("user_id", "unknown_user")
    query_id = str(uuid.uuid4()).replace("-", "")

    query_to_store = QueryDBModel(
        user_id=user_id,
        query=query_body.query_api,
        query_string=query_body.query_string,
        hypotheses=query_body.hypotheses,
        constraints=query_body.constraints,
        candidates_number=query_body.candidates_number,
        query_mode=query_body.query_mode,
        query_id=query_id,
    )

    query_data = {
        "id": query_id,
        "user_id": user_id,
        "query": query_to_store.model_dump_json(),
        "response": query_response.model_dump_json(),
    }
    query_id_inserted = db.add_query(query_data)
    logger.info(f"User query inserted with ID: {query_id}")

    return query_id_inserted


async def _handle_query_creation(
    request: Request, downstream_url: str, body: bytes, user_data: dict, db: PostgresDatabase
) -> Response:
    """Handle query creation requests."""
    if body is None:
        return Request(content="Invalid body", status_code=status.HTTP_400_BAD_REQUEST)

    validated_body = QueryReqFromUpstream.model_validate(json.loads(body))
    query_module_request = QueryReqToDownstream(
        formula=parse_formula(validated_body.query_api.formula),
        variables=validated_body.query_api.variables,
        n_candidates=validated_body.candidates_number,
        search_algorithm="beam",
        search_parameters=SearchParameters(beam_size=10),
    )

    downstream_url = downstream_url.replace("query_mode", validated_body.query_mode)
    response = await forward_request(request, downstream_url, query_module_request)

    if response.body is None:
        return Response(content="Invalid response body", status_code=status.HTTP_400_BAD_REQUEST)

    validated_response = QueryRespFromDownstream.model_validate(json.loads(response.body))
    logger.info(f"Validated response: {json.loads(response.body)}")

    insert_user_query(user_data, validated_body, validated_response, db)
    return response


async def _handle_filter_constraint(request: Request, downstream_url: str, body: bytes) -> Response:
    """Handle query creation requests."""
    if body is None:
        return Request(content="Invalid body", status_code=status.HTTP_400_BAD_REQUEST)

    downstream_url = downstream_url.replace("query_mode", "simple").replace(
        "/query", "/filter_constraint"
    )

    logger.info(f"Sending request to {downstream_url}")

    return await forward_request(request, downstream_url, None)


async def _handle_get_query(query_id: str, db: PostgresDatabase) -> Response:
    """Handle get query by ID requests."""
    validated_body = QueryId(id=query_id)
    query_data = get_query(validated_body, db)

    if query_data is not None:
        logger.info(f"Retrieved query data: {query_data['query']}")
        validated_response = QueryDBModel.model_validate(query_data["query"])
        return Response(content=validated_response.model_dump_json(), status_code=200)
    return Response(content="No query found", status_code=404)


async def _handle_get_user_queries(user_data: dict, db: PostgresDatabase) -> Response:
    """Handle get all user queries requests."""
    query_data = get_user_queries(user_data, db)
    if query_data is not None:
        logger.info(f"Retrieved query data: {query_data}")
        validated_response = QueryRespToUpstream.model_validate(
            {"queries": [row["query"] for row in query_data]}
        )
        return Response(content=validated_response.model_dump_json(), status_code=200)
    return Response(content="No queries found", status_code=404)


async def _handle_delete_query(query_id: str, user_data: dict, db: PostgresDatabase) -> Response:
    """Handle delete query requests."""
    if query_id is None:
        return Request(content="Invalid query_id", status_code=status.HTTP_400_BAD_REQUEST)

    validated_body = QueryId(id=query_id)
    logger.info(f"Deleting query with ID: {validated_body.id}")
    deleted_query = delete_query(validated_body.id, user_data["user_id"], db)
    if deleted_query:
        return Response(content="Query deleted", status_code=status.HTTP_200_OK)
    return Response(content="No query found to delete", status_code=status.HTTP_404_NOT_FOUND)


async def _handle_delete_user_queries(user_data: dict, db: PostgresDatabase) -> Response:
    """Handle delete all user queries requests."""
    logger.info(f"Deleting all queries for user: {user_data['email']}")
    deleted_queries = delete_all_user_queries(user_data["user_id"], db)
    if deleted_queries:
        return Response(content="All user queries deleted", status_code=status.HTTP_200_OK)
    return Response(
        content="No queries found to delete for user", status_code=status.HTTP_404_NOT_FOUND
    )


async def handle_query_service_request(
    request: Request, downstream_url: str, user_data: dict, db: PostgresDatabase
) -> Response:
    """
    Handle requests specific to the 'query' service.

    Args:
        request (Request): The incoming FastAPI request
        downstream_url (str): The URL of the downstream service
        user_data: Name and email of the user
        db: Connection to the DB to store the queries

    Returns:
        Response: The response from the downstream service
    """

    path = request.path_params.get("path").split("/")[0] or ""
    body = await request.body()
    # # Get the method (e.g., "GET", "POST")
    http_method = request.method
    
    logger.info(f"Method: {http_method} for path: {path}")
    logger.info(f"Handling query service request for path: {path}, {request.path_params}")


    match path:
        case "getquery":
            if http_method != "GET":
                return Response(
                    content="Invalid HTTP method for getquery", status_code=status.HTTP_405_METHOD_NOT_ALLOWED
                )
            query_id = request.path_params.get("path").split("/")[1] or ""
            response = await _handle_get_query(query_id, db)
        case "getuserqueries":
            if http_method != "GET":
                return Response(
                    content="Invalid HTTP method for getquery", status_code=status.HTTP_405_METHOD_NOT_ALLOWED
                )
            response = await _handle_get_user_queries(user_data, db)
        case "deletequery":
            if http_method != "DELETE":
                return Response(
                    content="Invalid HTTP method for getquery", status_code=status.HTTP_405_METHOD_NOT_ALLOWED
                )
            query_id = request.path_params.get("path").split("/")[1] or ""
            response = await _handle_delete_query(query_id, user_data, db)
        case "deleteuserqueries":
            if http_method != "DELETE":
                return Response(
                    content="Invalid HTTP method for getquery", status_code=status.HTTP_405_METHOD_NOT_ALLOWED
                )
            response = await _handle_delete_user_queries(user_data, db)
        case "filter_constraint":
            response = await _handle_filter_constraint(request, downstream_url, body)
        case "" | None:
            response = await _handle_query_creation(request, downstream_url, body, user_data, db)
        case _:
            # Handle Invalid Paths
            response = Response(
                content=f"Invalid path: {path}", status_code=status.HTTP_400_BAD_REQUEST
            )

    return response
