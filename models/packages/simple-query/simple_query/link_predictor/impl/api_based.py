from typing import TYPE_CHECKING

import requests

from simple_query.link_predictor.base import LinkPredictor
from simple_query.link_predictor.entities.configs import ApiBasedLinkPredictorConfig
from simple_query.link_predictor.values.errors import LinkPredictorError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from simple_query.link_predictor.entities.inputs import LinkPredictorInputTriple


class ApiBasedLinkPredictor(LinkPredictor[ApiBasedLinkPredictorConfig]):
    def _predict(self, triples: "Sequence[LinkPredictorInputTriple]") -> list[float]:
        relations_in_batch = list({t.relation_identifier for t in triples})
        request_data: dict[str, list[str]] = {
            "subject_id_list": [t.subject_identifier for t in triples],
            "relation_id_list": relations_in_batch,
            "object_id_list": [t.object_identifier for t in triples],
        }

        response = requests.post(
            url=self.config.url,
            json={"request": request_data},
        )
        if response.status_code != 200:  # noqa: PLR2004
            raise LinkPredictorError(
                f"link prediction failed with status code {response.status_code}"
            )

        try:
            response_json = response.json()
            relations_ids = response_json["relations_ids"]
            relations_probs = response_json["relations_probs"]
        except requests.exceptions.JSONDecodeError as e:
            raise LinkPredictorError(f"failed to decode JSON response: {response.content!r}") from e
        except KeyError as e:
            raise LinkPredictorError(f"response JSON does not have the key {e}") from e

        triple_probs: list[float] = []
        for i, triple in enumerate(triples):
            triple_probs.append(relations_probs[i][relations_ids.index(triple.relation_identifier)])

        return triple_probs
