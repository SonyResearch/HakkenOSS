from __future__ import annotations

from typing import TYPE_CHECKING

from filtering.core.entities.config.knowledge_graph import (
    KnowledgeGraphConfigBase,
    Neo4jKnowledgeGraphConfig,
    NetworkXKnowledgeGraphConfig,
)
from filtering.core.entities.config.node_filtering import (
    EntropyNodeFilteringConfig,
    NodeFilteringConfigBase,
    RandomNodeFilteringConfig,
    RecencyNodeFilteringConfig,
)
from filtering.core.entities.config.triple_filtering import RandomTripleFilteringConfig
from filtering.impl.kg import Neo4jKnowledgeGraph, NetworkXKnowledgeGraph
from filtering.impl.node_filtering import (
    EntropyNodeFiltering,
    RandomNodeFiltering,
    RecencyNodeFiltering,
)
from filtering.impl.triple_filtering import RandomTripleFiltering

if TYPE_CHECKING:
    from collections.abc import Mapping

    from filtering.core.contracts.knowledge_graph import KnowledgeGraph
    from filtering.core.contracts.node_filtering import NodeFiltering
    from filtering.core.contracts.triple_filtering import TripleFiltering
    from filtering.core.entities.config.triple_filtering import TripleFilteringConfigBase

KG_CLASS_MAPPING: Mapping[type[KnowledgeGraphConfigBase], type[KnowledgeGraph]] = {
    NetworkXKnowledgeGraphConfig: NetworkXKnowledgeGraph,
    Neo4jKnowledgeGraphConfig: Neo4jKnowledgeGraph,
}

NODE_FILTERING_CLASS_MAPPING: Mapping[type[NodeFilteringConfigBase], type[NodeFiltering]] = {
    EntropyNodeFilteringConfig: EntropyNodeFiltering,
    RecencyNodeFilteringConfig: RecencyNodeFiltering,
    RandomNodeFilteringConfig: RandomNodeFiltering,
}

TRIPLE_FILTERING_CLASS_MAPPING: Mapping[type[TripleFilteringConfigBase], type[TripleFiltering]] = {
    RandomTripleFilteringConfig: RandomTripleFiltering,
}
