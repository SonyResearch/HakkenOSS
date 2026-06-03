from typing import Any

import pytest
import yaml
from dotenv import load_dotenv
from query_common.parse.impl.lark import LarkParser
from query_common.values.types import ParserType

from simple_query.api.container import ApiConfig, SimpleQueryingContainer
from simple_query.kg.entities.configs import Neo4jKnowledgeGraphConfig
from simple_query.kg.impl.neo4j import Neo4jKnowledgeGraph
from simple_query.link_predictor.entities.configs import ApiBasedLinkPredictorConfig
from simple_query.link_predictor.impl.api_based import ApiBasedLinkPredictor
from simple_query.query.entities.configs import SimpleQueryingConfig
from simple_query.query.impl.simple import SimpleQuerying


@pytest.fixture
def kg_config_dict() -> dict[str, Any]:
    config_dict: dict[str, Any] = {
        "config_type": "neo4j",
        "username": "your_username",
        "password": "your_password",
        "use_okta": False,
    }
    return config_dict


@pytest.fixture
def link_predictor_config_dict() -> dict[str, Any]:
    config_dict: dict[str, Any] = {
        "config_type": "api_based",
        "url": "http://link_predictor_url",
    }
    return config_dict


@pytest.fixture
def querying_config_dict() -> dict[str, Any]:
    config_dict: dict[str, Any] = {"config_type": "simple"}
    return config_dict


def test_api_config(kg_config_dict, link_predictor_config_dict, querying_config_dict):
    api_config = ApiConfig(
        kg_config=kg_config_dict,
        link_predictor_config=link_predictor_config_dict,
        querying_config=querying_config_dict,
        parser_type=ParserType.LARK,
    )
    assert isinstance(api_config.kg_config, Neo4jKnowledgeGraphConfig)
    assert isinstance(api_config.link_predictor_config, ApiBasedLinkPredictorConfig)
    assert isinstance(api_config.querying_config, SimpleQueryingConfig)
    assert api_config.parser_type == ParserType.LARK


def test_api_config_from_envfile(envfile_path):
    load_dotenv(envfile_path, override=True)
    api_config = ApiConfig()
    assert isinstance(api_config.kg_config, Neo4jKnowledgeGraphConfig)
    assert isinstance(api_config.link_predictor_config, ApiBasedLinkPredictorConfig)
    assert isinstance(api_config.querying_config, SimpleQueryingConfig)


def test_api_config_from_yaml(yaml_path):
    with open(yaml_path) as f:
        yaml_config = yaml.safe_load(f)
    api_config = ApiConfig.model_validate(yaml_config, strict=True)
    assert isinstance(api_config.kg_config, Neo4jKnowledgeGraphConfig)
    assert isinstance(api_config.link_predictor_config, ApiBasedLinkPredictorConfig)
    assert isinstance(api_config.querying_config, SimpleQueryingConfig)


def test_container(kg_config_dict, link_predictor_config_dict, querying_config_dict):
    api_config = ApiConfig(
        kg_config=kg_config_dict,
        link_predictor_config=link_predictor_config_dict,
        querying_config=querying_config_dict,
        parser_type=ParserType.LARK,
    )
    container = SimpleQueryingContainer()
    container.config.from_pydantic(api_config)

    assert isinstance(container.kg(), Neo4jKnowledgeGraph)
    assert isinstance(container.link_predictor(), ApiBasedLinkPredictor)
    assert isinstance(container.querying(), SimpleQuerying)
    assert isinstance(container.parser(), LarkParser)
