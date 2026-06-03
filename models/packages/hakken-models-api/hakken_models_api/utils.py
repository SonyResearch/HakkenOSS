import json
from http import HTTPStatus
from typing import cast

import requests
import torch
from loguru import logger
from torch import Tensor


def build_entity_pairs_tensor(
    subject_list: list[int], object_list: list[int], device: str
) -> Tensor:
    subjects_pt = torch.tensor(subject_list, device=device)
    objects_pt = torch.tensor(object_list, device=device)

    return torch.stack((subjects_pt, objects_pt), dim=1)


def create_request(url: str, data: dict) -> dict | None:
    # Convert the data to JSON
    json_data = json.dumps(data)

    # Set the headers
    headers = {"Content-Type": "application/json"}

    # Make the POST request
    response = requests.post(url, data=json_data, headers=headers)

    # Check the response
    if response.status_code == HTTPStatus.OK:
        # Request was successful
        result = response.json()
        logger.info(f"Prediction result: {result}")
        return cast(dict, result)
    # There was an error
    logger.error(f"Error: {response.status_code} {response.text}")
    return None
