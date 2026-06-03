from typing import TYPE_CHECKING

import numpy as np
import requests
from pydantic import BaseModel
from query_common.entities.kg.identifier import ConceptIdentifier, RelationIdentifier
from query_common.entities.kg.triple import Triple

from complex_query.core.contracts.link_predictor import LinkPredictor
from complex_query.core.entities.config.link_predictor import ApiBasedLinkPredictorConfig
from complex_query.core.values.errors import PredictorLogicError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray
    from query_common.entities.kg.triple import Triple


class LinkPredictorApiInput(BaseModel):
    subject_id_list: list[ConceptIdentifier]
    object_id_list: list[ConceptIdentifier]
    relation_id_list: list[RelationIdentifier]


class LinkPredictorApiOutput(BaseModel):
    relations_ids: list[RelationIdentifier]
    relations_probs: list[list[float]]
    relations_scores: list[list[float]]

    def get_prob(self, ith_triple: int, relation_identifier: str) -> float:
        return self.relations_probs[ith_triple][self.relations_ids.index(relation_identifier)]


MAX_CHAR_ERROR_DETAIL = 300


class ApiBasedLinkPredictor(LinkPredictor[ApiBasedLinkPredictorConfig]):
    def call_api(
        self,
        data: LinkPredictorApiInput,
    ) -> LinkPredictorApiOutput:
        headers = {"Content-Type": "application/json"}
        response = requests.post(
            self.config.url, json={"request": data.model_dump()}, headers=headers
        )
        if response.status_code != 200:  # noqa: PLR2004
            response_detail = str(response.json()["detail"])
            if len(response_detail) > MAX_CHAR_ERROR_DETAIL:
                response_detail = (
                    response_detail[:MAX_CHAR_ERROR_DETAIL]
                    + f"(... details cut after {MAX_CHAR_ERROR_DETAIL} characters)"
                )
            raise PredictorLogicError(
                f"Error when predicting data on the core model API. Detail: {response_detail}"
            )
        return LinkPredictorApiOutput(**response.json())

    def predict(self, triple: "Triple") -> float:
        data = LinkPredictorApiInput(
            subject_id_list=[triple.subject_identifier],
            object_id_list=[triple.object_identifier],
            relation_id_list=[triple.relation_identifier],
        )
        output = self.call_api(data)
        return output.get_prob(0, triple.relation_identifier)

    def predict_batch(self, triples: "Sequence[Triple]") -> "NDArray[np.float64]":
        relations_in_batch = list({t.relation_identifier for t in triples})
        data = LinkPredictorApiInput(
            subject_id_list=[t.subject_identifier for t in triples],
            object_id_list=[t.object_identifier for t in triples],
            relation_id_list=relations_in_batch,
        )
        output = self.call_api(data)
        scores = [
            output.get_prob(i, triple.relation_identifier) for i, triple in enumerate(triples)
        ]
        return np.array(scores)
