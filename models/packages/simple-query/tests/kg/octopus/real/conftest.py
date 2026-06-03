import pytest

from simple_query.kg.entities.configs import Neo4jKnowledgeGraphConfig
from simple_query.kg.impl.neo4j import Neo4jKnowledgeGraph


@pytest.fixture(scope="session")
def neo4j_kg() -> Neo4jKnowledgeGraph:
    config = Neo4jKnowledgeGraphConfig(use_okta=True)
    return Neo4jKnowledgeGraph(config)
