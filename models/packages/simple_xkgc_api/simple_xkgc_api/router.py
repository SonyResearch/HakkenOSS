from typing import TYPE_CHECKING, Protocol

from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, HTTPException
from hakken_ml_toolkit.ml_base_structures import Fact
from spaice_inference_api import ILogger, LoggerToken, ModelToken

from simple_xkgc_api.container import Container
from simple_xkgc_api.entities.config import APIConfig
from simple_xkgc_api.entities.request import (
    PathExplainerLengthRequest,
    PathExplainerRequest,
)
from simple_xkgc_api.entities.response import (
    ExplanationAPI,
    PathExplainerLengthResponse,
    PathExplainerResponse,
)

if TYPE_CHECKING:
    import pandas as pd


class PathExplainer(Protocol):
    def explanation_len(self, triple_to_probe: Fact) -> int: ...
    def explain(self, triple_to_probe: Fact, **kwargs: object) -> "pd.DataFrame": ...


TRIPLE_SIZE = 3


class TripleSizeError(ValueError):
    def __init__(self, size: int) -> None:
        super().__init__(f"Each triple must contain exactly {size} string elements")


def _fact_from_list(triple_to_probe_as_list: list[str]) -> Fact:
    if len(triple_to_probe_as_list) != TRIPLE_SIZE:
        raise TripleSizeError(TRIPLE_SIZE)

    subject, relation, object_id = triple_to_probe_as_list
    return (subject, relation, object_id)


router = APIRouter()


@router.post("/path_explainer/explain", response_model=PathExplainerResponse)
@inject
def explain(
    request: PathExplainerRequest,
    logger: ILogger = Depends(Provide[LoggerToken]),
    explainer: PathExplainer = Depends(Provide[ModelToken]),
    config: APIConfig = Depends(Provide[Container.config]),
) -> PathExplainerResponse:
    logger.debug(f"Input: {request}")

    try:
        explanations = {}

        for triple_to_probe_as_list in request.triples_to_probe:
            triple_to_probe = _fact_from_list(triple_to_probe_as_list)
            length = explainer.explanation_len(triple_to_probe)
            if (
                request.min_explanation_length is not None
                and length < request.min_explanation_length
            ):
                continue
            logger.info("Computing explanations..")

            df_expl_all: pd.DataFrame = explainer.explain(
                triple_to_probe,
                device=config.run.device,
                explanation_type_list=request.explanation_configs,
                rerank_strategy=request.rerank_strategy,
                allowed_relations_ids=request.allowed_relation_ids,
            )

            df_expl = df_expl_all.head(request.num_explanations)

            logger.info(f"..found {len(df_expl)} explanations")

            explanation_list = []
            for explanation_data_i in df_expl[["score", "explanation"]].to_dict("records"):
                explanation_i = ExplanationAPI(
                    data=explanation_data_i["explanation"],
                    length=length,
                    score=explanation_data_i["score"],
                )
                explanation_list.append(explanation_i)

            explanations[str(triple_to_probe)] = explanation_list

        return PathExplainerResponse(explanations=explanations)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/path_explainer/explanation_length", response_model=PathExplainerLengthResponse)
@inject
def explanation_length(
    request: PathExplainerLengthRequest,
    logger: ILogger = Depends(Provide[LoggerToken]),
    explainer: PathExplainer = Depends(Provide[ModelToken]),
) -> PathExplainerLengthResponse:
    logger.debug(f"Input: {request}")

    try:
        length_dict = {}

        for triple_to_probe_as_list in request.triples_to_probe:
            triple_to_probe = _fact_from_list(triple_to_probe_as_list)

            length = explainer.explanation_len(triple_to_probe)

            length_dict[str(triple_to_probe)] = length

        return PathExplainerLengthResponse(length_dict=length_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
