from typing import Any

from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, HTTPException
from query_common.entities.query import QueryRequest, QueryResponse
from spaice_inference_api import ILogger, LoggerToken

from complex_query.core.actions import answer_query
from complex_query.core.values.errors import InputError, LogicError

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
@inject
def query(
    request: QueryRequest,
    logger: ILogger = Depends(Provide[LoggerToken]),  # noqa: B008
) -> Any:
    logger.info(f"Received request with formula: {request.formula}")
    logger.debug(f"Input: {request.model_dump()}")
    try:
        res = answer_query(request)
    except InputError as e:
        logger.exception(f"Error: {e}")
        raise HTTPException(status_code=422, detail=str(e)) from e
    except (LogicError, Exception) as e:
        logger.exception(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
    return res
