import sys
from pathlib import Path

import pydantic_yaml
from loguru import logger

from complex_query.container import QueryingContainer, QueryingSettings


def wire_container(module_name):
    container = QueryingContainer()
    config_root = Path(__file__).parent
    config = pydantic_yaml.parse_yaml_file_as(QueryingSettings, config_root / "config.yaml")
    container.config.from_pydantic(config)
    container.init_resources()
    container.wiring_config.packages.extend(["complex_query"])
    container.wiring_config.modules.extend([sys.modules[module_name]])
    container.wire()
    logger.info("Container bootstrapped")
