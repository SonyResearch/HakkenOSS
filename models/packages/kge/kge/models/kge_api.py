import json
import os
from typing import Any, cast

import requests
from hakken_ml_toolkit.ml_utils.extras import TensorCreator

from kge.common.types import FloatTensor2D, LongTensor2D

SUCCESS_CODE = 200


class KGEAPI:
    def __init__(self, base_url: str = "http://localhost:8088/test/kge"):
        self.base_url = base_url

    def request(self, endpoint: str, data: Any) -> dict:
        json_data = json.dumps(data)

        headers = {"Content-Type": "application/json"}

        url = os.path.join(self.base_url, endpoint)
        response = requests.post(url, data=json_data, headers=headers)

        if response.status_code == SUCCESS_CODE:
            return cast("dict", response.json())

        msg = f"{response.status_code} Client Error: {response.reason} for url: {url}"
        raise requests.exceptions.HTTPError(msg, response=response)

    def score(self, sro_batch: LongTensor2D, normalize: bool = False) -> FloatTensor2D:
        device = sro_batch.device

        triple_list = sro_batch.tolist()

        response = self.request(
            "score",
            {"request": {"triple_index_list": triple_list, "normalize": normalize}},
        )

        score_list = response["scores_list"]

        return TensorCreator.float_tensor(score_list, device=device)

    def fit_score_scaler(self, loader_kwargs: dict | None = None, overwrite: bool = False) -> bool:
        response = self.request(
            "fit_score_scaler",
            {"request": {"loader_kwargs": loader_kwargs, "overwrite": overwrite}},
        )

        return cast("bool", response["success"])
