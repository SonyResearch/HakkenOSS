import uuid

from dependency_injector.wiring import Provide, inject
from fastapi import Depends, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import HTTPException

from spaice_inference_api.config import Settings, SettingsToken
from spaice_inference_api.core.contract.logger import ILogger, LoggerToken
from spaice_inference_api.core.value.request import RequestIdHeader

AUTH_HEADER_PARTS = 2
REQUEST_LOGGER_DEPENDENCY = Depends(Provide[LoggerToken])


@inject
async def authentication_middleware(
    request: Request,
    call_next,
    settings: Settings = Provide[SettingsToken],
    logger: ILogger = Provide[LoggerToken],
):
    # TODO: create regex to capture the pathss
    public_paths = [
        "/health-check",
        f"/{settings.SPAICE_MODEL_NAME}/health-check",
        "/metrics",
        f"/{settings.SPAICE_MODEL_NAME}/metrics",
    ]

    if settings.SPAICE_INFERENCE_API_KEY is not None and request.url.path not in public_paths:
        authentication_header = request.headers.get("Authentication", "Bearer ")
        if len(authentication_header.split(" ")) != AUTH_HEADER_PARTS:
            # We have to manually call this handler cause
            # exception handler is not called for middlewares
            return await http_exception_handler(
                request,
                HTTPException(
                    403,
                    'Invalid api key header format: Valid format is "Bearer <API_KEY>"',
                ),
            )

        _, token = authentication_header.split(" ")
        if token != settings.SPAICE_INFERENCE_API_KEY:
            logger.warning(f'Requested endpoint "{request.url.path}" with invalid api key')
            return await http_exception_handler(request, HTTPException(403, "Invalid api key"))

    return await call_next(request)


@inject
async def request_id_middleware(
    request: Request, call_next, logger: ILogger = REQUEST_LOGGER_DEPENDENCY
):
    if RequestIdHeader in request.headers:
        request_id = request.headers[RequestIdHeader]
    else:
        request_id = str(uuid.uuid4())
        # Here we want to set the request id so that it's
        # always present. Headers are immutable by default
        # so this is a way to inject the request id
        request.headers.__dict__["_list"].append(
            (
                RequestIdHeader.lower().encode(),
                request_id.encode(),
            )
        )
    logger.set_labels({"request_id": request_id})

    response = await call_next(request)
    response.headers[RequestIdHeader] = request_id
    return response
