from typing import Any

import torch
from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, HTTPException
from kge.common.actions.kge_inference_action import KGEInferenceActions
from kge.common.entities import (
    KGEPredictRequest,
    KGEPredictResponse,
    KGEResponse,
    KGEScoreIndexRequest,
    KGEScoreRequest,
    KGEScoreResponse,
    KGESFitScoreScalerRequest,
)
from spaice_inference_api import ILogger, LoggerToken, ModelToken

from kge_api.config import APIConfig
from kge_api.container import Container
from kge_api.kge_loader import KGEExperimentData, prepare_score_scaler

router = APIRouter()


@router.post("/kge/predict", response_model=KGEPredictResponse)
@inject
def predict(
    request: KGEPredictRequest,
    logger: ILogger = Depends(Provide[LoggerToken]),
    experiment_data: KGEExperimentData = Depends(Provide[ModelToken]),
    config: APIConfig = Provide[Container.config],
) -> Any:
    logger.debug(f"Input: {request}")

    try:
        res = KGEInferenceActions.predict(
            request=request,
            kge=experiment_data.model,
            data_processing=experiment_data.data_processor,
            device=config.device,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return res


@router.post("/kge/score", response_model=KGEScoreResponse)
@inject
@torch.no_grad()
def score(
    request: KGEScoreRequest,
    logger: ILogger = Depends(Provide[LoggerToken]),
    experiment_data: KGEExperimentData = Depends(Provide[ModelToken]),
    config: APIConfig = Provide[Container.config],
) -> Any:
    logger.debug(f"Input: {request}")

    try:
        res = KGEInferenceActions.score(
            request=request,
            kge=experiment_data.model,
            data_processing=experiment_data.data_processor,
            device=config.device,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return res


@router.post("/kge/score_from_index", response_model=KGEScoreResponse)
@inject
@torch.no_grad()
def score_from_index(
    request: KGEScoreIndexRequest,
    logger: ILogger = Depends(Provide[LoggerToken]),
    experiment_data: KGEExperimentData = Depends(Provide[ModelToken]),
    config: APIConfig = Provide[Container.config],
) -> Any:
    logger.debug(f"Input: {request}")

    try:
        res = KGEInferenceActions.score_from_index(
            request=request,
            kge=experiment_data.model,
            device=config.device,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return res


@router.post("/kge/fit_score_scaler", response_model=KGEResponse)
@inject
def fit_score_scaler(
    request: KGESFitScoreScalerRequest,
    _logger: ILogger = Depends(Provide[LoggerToken]),
    experiment_data: KGEExperimentData = Depends(Provide[ModelToken]),
    config: APIConfig = Provide[Container.config],
) -> Any:
    try:
        prepare_score_scaler(
            kge_bundle=experiment_data,
            score_scaler_json_path=config.score_scaler_json_path,
            overwrite=request.overwrite,
            device=config.device,
            loader_kwargs=request.loader_kwargs,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return KGEResponse(success=True, message="Score scaler fitted")


@router.post("/kge/device")
@inject
def device(
    _logger: ILogger = Depends(Provide[LoggerToken]),
    experiment_data: KGEExperimentData = Depends(Provide[ModelToken]),
) -> Any:
    try:
        res = {str(param.device) for param in experiment_data.model.parameters()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return res
