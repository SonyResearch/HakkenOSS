import argparse
import json
import sys

from loguru import logger
from query_common.entities.query import QueryRequest
from spaice_inference_api import Container

from complex_query.core.actions import answer_query


def wire_container():
    container = Container()
    container.init_resources()
    container.wiring_config.packages.extend(["complex_query"])
    container.wiring_config.modules.extend([sys.modules[__name__]])
    container.wire()
    logger.info("Container bootstrapped")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--request-json",
        type=str,
        required=True,
        help="Path to a json file with the request.",
    )
    args = parser.parse_args()
    with open(args.request_json) as file:
        request = QueryRequest.model_validate(json.load(file))
    wire_container()
    res = answer_query(request)
