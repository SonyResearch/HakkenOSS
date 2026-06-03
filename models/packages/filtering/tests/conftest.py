from pathlib import Path

import pytest
import yaml

from filtering.core.entities.config.knowledge_graph import Neo4jKnowledgeGraphConfig


@pytest.fixture
def test_data_root():
    return (Path(__file__).parent / "test_data").absolute()


@pytest.fixture
def nodes_path(test_data_root):
    return test_data_root / "nodes.csv"


@pytest.fixture
def edges_path(test_data_root):
    return test_data_root / "edges.csv"


@pytest.fixture
def edge_cache_pickle_directory(test_data_root):
    return test_data_root / "edge_cache" / "pickle"


@pytest.fixture
def edge_cache_ndjson_directory(test_data_root):
    return test_data_root / "edge_cache" / "ndjson"


@pytest.fixture
def container_config_env_path(test_data_root):
    return test_data_root / "test_container_config.env"


@pytest.fixture
def neo4j_kg_config_yaml_path(test_data_root):
    return test_data_root / "test_neo4j_kg_config.yaml"


@pytest.fixture
def neo4j_kg_config(neo4j_kg_config_yaml_path):
    with open(neo4j_kg_config_yaml_path) as f:
        config_dict = yaml.safe_load(f)

    return Neo4jKnowledgeGraphConfig.model_validate(config_dict)
