"""THiGER API router."""

import traceback
from typing import Any

import torch
from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, HTTPException
from hakken_models.core.configs import THiGERInferenceConfig
from hakken_models.core.constants import MissingPolicy
from hakken_models.models.thiger import THiGERArtifacts
from hakken_models.models.thiger.actions import thiger_predict, thiger_score
from spaice_inference_api import ILogger, LoggerToken, ModelToken

from hakken_models_api.config import HakkenModelsAPIConfig
from hakken_models_api.container import Container
from hakken_models_api.entities.data import (
    EntityPairIndexRequest,
    EntityPairIndexResponse,
    FactIndexRequest,
    FactIndexResponse,
    SampleFactsRequest,
    SampleFactsResponse,
)
from hakken_models_api.entities.predict import ModelPredictRequest, ModelPredictResponse
from hakken_models_api.entities.score import ModelScoreRequest, ModelScoreResponse
from hakken_models_api.utils import build_entity_pairs_tensor

router = APIRouter(prefix="/thiger")


@router.post("/predict", response_model=ModelPredictResponse)
@inject
def predict(
    request: ModelPredictRequest,
    logger: ILogger = Depends(Provide[LoggerToken]),
    artifacts: THiGERArtifacts = Depends(Provide[ModelToken]),
    config: HakkenModelsAPIConfig = Provide[Container.config],
) -> Any:
    logger.debug(f"Input: {request}")
    dataset = artifacts.dataset
    if dataset is None:
        raise HTTPException(status_code=500, detail="Dataset is not loaded in THiGER artifacts.")

    try:
        inference_config = THiGERInferenceConfig(**(request.inference_config or {}))
        thiger = artifacts.thiger
        subject_index_list = dataset.map_node_ids_to_indexes(
            request.subject_id_list, on_missing=inference_config.on_missing
        )

        object_index_list = dataset.map_node_ids_to_indexes(
            request.object_id_list, on_missing=inference_config.on_missing
        )

        entity_pairs = build_entity_pairs_tensor(
            subject_index_list, object_index_list, device=config.device
        )

        logits_cols = None
        if request.relation_id_list is not None:
            relations_ids = request.relation_id_list

        else:
            relations_ids = dataset.get_relation_ids()
        logits_cols = dataset.map_relation_ids_to_indexes(
            relations_ids, on_missing=MissingPolicy.RAISE
        )

        relation_prediction = thiger_predict(
            entity_pairs=entity_pairs,
            thiger=thiger,
            dataset=dataset,
            config=inference_config,
            logits_cols=logits_cols,
        )

        res = ModelPredictResponse(
            relations_ids=relations_ids,
            relations_probs=relation_prediction.probs,
            relations_scores=relation_prediction.logits,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return res


@router.post("/score", response_model=ModelScoreResponse)
@inject
@torch.no_grad()
def score(
    request: ModelScoreRequest,
    logger: ILogger = Depends(Provide[LoggerToken]),
    artifacts: THiGERArtifacts = Depends(Provide[ModelToken]),
    config: HakkenModelsAPIConfig = Provide[Container.config],
) -> ModelScoreResponse:
    logger.debug(f"Input: {request}")

    dataset = artifacts.dataset
    if dataset is None:
        raise HTTPException(status_code=500, detail="Dataset is not loaded in THiGER artifacts.")

    try:
        thiger = artifacts.thiger
        inference_config = THiGERInferenceConfig(**(request.inference_config or {}))
        facts_list = dataset.map_fact_ids_to_indexes(
            request.facts_list, on_missing=inference_config.on_missing
        )

        facts_pt = torch.tensor(facts_list, device=config.device)

        relation_prediction = thiger_score(
            facts=facts_pt,
            thiger=thiger,
            dataset=dataset,
            config=inference_config,
        )

        res = ModelScoreResponse(
            scores_list=relation_prediction.logits,
            normalized_scores_list=relation_prediction.probs,
        )

    except Exception as e:
        tb = traceback.format_exc()
        logger.exception(tb)
        raise HTTPException(status_code=500, detail={"error": str(e), "traceback": tb}) from e

    return res


@router.post("/entity-pair-indexes", response_model=EntityPairIndexResponse)
@inject
def get_entity_pair_indexes(
    request: EntityPairIndexRequest,
    logger: ILogger = Depends(Provide[LoggerToken]),
    artifacts: THiGERArtifacts = Depends(Provide[ModelToken]),
) -> Any:
    logger.debug(f"Input: {request}")

    dataset = artifacts.dataset
    if dataset is None:
        raise HTTPException(status_code=500, detail="Dataset is not loaded in THiGER artifacts.")

    try:
        inference_config = THiGERInferenceConfig(**(request.inference_config or {}))

        subject_index_list = dataset.map_node_ids_to_indexes(
            request.subject_id_list,
            on_missing=inference_config.on_missing,
        )

        object_index_list = dataset.map_node_ids_to_indexes(
            request.object_id_list,
            on_missing=inference_config.on_missing,
        )

        # build entity pair tensor (2, N)
        entity_pairs = build_entity_pairs_tensor(
            subject_index_list,
            object_index_list,
            device="cpu",  # indexes only, keep on CPU
        )

        return EntityPairIndexResponse(
            subject_index_list=subject_index_list,
            object_index_list=object_index_list,
            entity_pairs=entity_pairs.tolist(),
        )

    except Exception as e:
        tb = traceback.format_exc()
        logger.exception(tb)
        raise HTTPException(
            status_code=500,
            detail={"error": str(e), "traceback": tb},
        ) from e


@router.post("/fact-indexes", response_model=FactIndexResponse)
@inject
def get_fact_indexes(
    request: FactIndexRequest,
    logger: ILogger = Depends(Provide[LoggerToken]),
    artifacts: THiGERArtifacts = Depends(Provide[ModelToken]),
) -> Any:
    logger.debug(f"Input: {request}")

    dataset = artifacts.dataset
    if dataset is None:
        raise HTTPException(status_code=500, detail="Dataset is not loaded in THiGER artifacts.")

    try:
        inference_config = THiGERInferenceConfig(**(request.inference_config or {}))

        fact_index_list = dataset.map_fact_ids_to_indexes(
            request.facts_list,
            on_missing=inference_config.on_missing,
        )

        return FactIndexResponse(
            fact_index_list=fact_index_list,
        )

    except Exception as e:
        tb = traceback.format_exc()
        logger.exception(tb)
        raise HTTPException(
            status_code=500,
            detail={"error": str(e), "traceback": tb},
        ) from e


@router.post("/sample-facts", response_model=SampleFactsResponse)
@inject
def sample_random_facts(
    request: SampleFactsRequest,
    logger: ILogger = Depends(Provide[LoggerToken]),
    artifacts: THiGERArtifacts = Depends(Provide[ModelToken]),
) -> Any:
    """
    Sample random facts (triples) from the dataset splits.
    """
    logger.debug(f"Input: {request}")
    dataset = artifacts.dataset
    if dataset is None:
        raise HTTPException(status_code=500, detail="Dataset is not loaded in THiGER artifacts.")

    try:
        result = dataset.sample_random_facts(
            splits=request.splits,
            num_samples=request.num_samples,
        )
        return SampleFactsResponse(facts_list=result)

    except HTTPException:
        raise

    except Exception as e:
        tb = traceback.format_exc()
        logger.exception(tb)
        raise HTTPException(status_code=500, detail={"error": str(e), "traceback": tb}) from e
