"""SeGAL API router."""

import traceback
from typing import Any

import torch
from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, HTTPException
from hakken_models.core.configs import SeGALInferenceConfig
from hakken_models.models.segal import SeGALArtifacts
from hakken_models.models.segal.actions import segal_score
from hakken_models.models.segal.text_scoring import segal_score_text
from spaice_inference_api import ILogger, LoggerToken, ModelToken

from hakken_models_api.config import HakkenModelsAPIConfig
from hakken_models_api.container import Container
from hakken_models_api.entities.data import (
    FactIndexRequest,
    FactIndexResponse,
    SampleFactsRequest,
    SampleFactsResponse,
)
from hakken_models_api.entities.score import ModelScoreRequest, ModelScoreResponse
from hakken_models_api.entities.segal import (
    ScoreTextRequest,
    ScoreTextResponse,
    SeGALInfoResponse,
)

router = APIRouter(prefix="/segal")


@router.get("/info", response_model=SeGALInfoResponse)
@inject
def get_info(
    logger: ILogger = Depends(Provide[LoggerToken]),
    artifacts: SeGALArtifacts = Depends(Provide[ModelToken]),
) -> Any:
    """Return dataset and model metadata for the loaded SeGAL model."""
    logger.debug("SeGAL info requested")
    dataset = artifacts.dataset
    if dataset is None:
        raise HTTPException(
            status_code=500,
            detail="Dataset is not loaded in SeGAL artifacts.",
        )

    return SeGALInfoResponse(
        num_entities=dataset.num_entities,
        num_relations=dataset.num_relations,
        has_embeddings=dataset.has_embeddings,
        embedding_dim=dataset.embedding_dim,
    )


@router.post("/score", response_model=ModelScoreResponse)
@inject
@torch.no_grad()
def score(
    request: ModelScoreRequest,
    logger: ILogger = Depends(Provide[LoggerToken]),
    artifacts: SeGALArtifacts = Depends(Provide[ModelToken]),
    config: HakkenModelsAPIConfig = Provide[Container.config],
) -> ModelScoreResponse:
    """Score facts (s, r, o triples) using the SeGAL model with temporal context."""
    logger.debug(f"Input: {request}")

    dataset = artifacts.dataset
    if dataset is None:
        raise HTTPException(
            status_code=500,
            detail="Dataset is not loaded in SeGAL artifacts.",
        )

    try:
        inference_config = SeGALInferenceConfig(**(request.inference_config or {}))
        fact_index_list = dataset.map_fact_ids_to_indexes(
            request.facts_list,
            on_missing=inference_config.on_missing,
        )

        facts_pt = torch.tensor(fact_index_list, device=config.device)

        relation_prediction = segal_score(
            facts=facts_pt,
            segal=artifacts.segal,
            dataset=dataset,
            config=inference_config,
        )

        return ModelScoreResponse(
            scores_list=relation_prediction.logits,
            normalized_scores_list=relation_prediction.probs,
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
    artifacts: SeGALArtifacts = Depends(Provide[ModelToken]),
) -> Any:
    """Map fact (subject, relation, object) string IDs to numeric indexes."""
    logger.debug(f"Input: {request}")
    dataset = artifacts.dataset
    if dataset is None:
        raise HTTPException(
            status_code=500,
            detail="Dataset is not loaded in SeGAL artifacts.",
        )

    try:
        inference_config = SeGALInferenceConfig(**(request.inference_config or {}))

        fact_index_list = dataset.map_fact_ids_to_indexes(
            request.facts_list,
            on_missing=inference_config.on_missing,
        )

        return FactIndexResponse(fact_index_list=fact_index_list)

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
    artifacts: SeGALArtifacts = Depends(Provide[ModelToken]),
) -> Any:
    """Sample random facts (triples) from the dataset splits."""
    logger.debug(f"Input: {request}")
    dataset = artifacts.dataset
    if dataset is None:
        raise HTTPException(
            status_code=500,
            detail="Dataset is not loaded in SeGAL artifacts.",
        )

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


@router.post("/score-text", response_model=ScoreTextResponse)
@inject
@torch.no_grad()
def score_text(
    request: ScoreTextRequest,
    logger: ILogger = Depends(Provide[LoggerToken]),
    artifacts: SeGALArtifacts = Depends(Provide[ModelToken]),
) -> ScoreTextResponse:
    """Score facts given as text. Entities/relations are encoded via the model embedder."""
    logger.debug(f"Input: {request}")

    try:
        inference_config = SeGALInferenceConfig(**(request.inference_config or {}))

        target_triples = [
            (f.subject, f.relation, f.object, f.timestamp) for f in request.target_facts
        ]
        context_triples = (
            [(f.subject, f.relation, f.object, f.timestamp) for f in request.context_facts]
            if request.context_facts
            else None
        )

        prediction = segal_score_text(
            target_facts=target_triples,
            segal=artifacts.segal,
            embedder=artifacts.embedder,
            context_facts=context_triples,
            config=inference_config,
        )

        return ScoreTextResponse(
            scores_list=prediction.logits,
            normalized_scores_list=prediction.probs,
        )

    except Exception as e:
        tb = traceback.format_exc()
        logger.exception(tb)
        raise HTTPException(
            status_code=500,
            detail={"error": str(e), "traceback": tb},
        ) from e
