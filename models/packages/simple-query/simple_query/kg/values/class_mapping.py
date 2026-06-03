from typing import TYPE_CHECKING

from simple_query.kg.entities.configs import (
    Neo4jKnowledgeGraphConfig,
)
from simple_query.kg.impl.neo4j import Neo4jKnowledgeGraph

if TYPE_CHECKING:
    from collections.abc import Mapping

    from simple_query.kg.base import KnowledgeGraph
    from simple_query.kg.entities.configs import KnowledgeGraphConfigBase


CLASS_MAPPING: "Mapping[type[KnowledgeGraphConfigBase], type[KnowledgeGraph]]" = {
    Neo4jKnowledgeGraphConfig: Neo4jKnowledgeGraph,
}
