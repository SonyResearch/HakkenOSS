from typing import Any

from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, HTTPException

from service_sample.entities import MyInferenceRequest, MyInferenceResponse
from service_sample.model import MyModel
from spaice_inference_api import ILogger, LoggerToken, ModelToken

router = APIRouter()
LOGGER_DEPENDENCY = Depends(Provide[LoggerToken])
MODEL_DEPENDENCY = Depends(Provide[ModelToken])


@router.post("/predict", response_model=MyInferenceResponse)
@inject
def predict(
    request: MyInferenceRequest,
    logger: ILogger = LOGGER_DEPENDENCY,
    model: MyModel = MODEL_DEPENDENCY,
) -> Any:
    logger.debug(f"Input: {request.model_dump()}")
    try:
        res = model.predict(request)
    except Exception as e:
        logger.exception(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

    # OK now I have to do the prediction
    return res
