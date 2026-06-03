import pytest
from dotenv import load_dotenv
from pydantic import TypeAdapter

from filtering.container import FilteringContainer, FilteringSettings
from filtering.core.entities.config.knowledge_graph import (
    NetworkXKnowledgeGraphConfig,
)
from filtering.core.entities.config.node_filtering import NodeFilteringConfig
from filtering.core.entities.config.triple_filtering import TripleFilteringConfig
from filtering.core.values.class_mapping import (
    KG_CLASS_MAPPING,
    NODE_FILTERING_CLASS_MAPPING,
    TRIPLE_FILTERING_CLASS_MAPPING,
)
from filtering.core.values.types import NodeFilteringType, TripleFilteringType
from filtering.impl.kg import NetworkXKnowledgeGraph
from filtering.impl.node_filtering import EntropyNodeFiltering
from filtering.impl.triple_filtering import RandomTripleFiltering


@pytest.fixture
def kg_config(nodes_path, edges_path):
    return NetworkXKnowledgeGraphConfig(nodes_path=nodes_path, edges_path=edges_path)


@pytest.fixture
def node_filtering_config(request):
    return TypeAdapter(NodeFilteringConfig).validate_python({"config_type": request.param})


@pytest.fixture
def triple_filtering_config(request):
    return TypeAdapter(TripleFilteringConfig).validate_python({"config_type": request.param})


class TestContainer:
    def test_from_env(self, container_config_env_path):
        load_dotenv(container_config_env_path, override=True)

        container = FilteringContainer()
        container.config.from_pydantic(FilteringSettings())

        container.wire(modules=[__name__], packages=["filtering"])

        assert isinstance(container.kg(), NetworkXKnowledgeGraph)
        assert isinstance(container.node_filtering(), EntropyNodeFiltering)
        assert isinstance(container.triple_filtering(), RandomTripleFiltering)

    @pytest.mark.parametrize("node_filtering_config", list(NodeFilteringType), indirect=True)
    @pytest.mark.parametrize("triple_filtering_config", list(TripleFilteringType), indirect=True)
    def test(self, kg_config, node_filtering_config, triple_filtering_config):
        config = FilteringSettings(
            kg_config=kg_config,
            node_filtering_config=node_filtering_config,
            triple_filtering_config=triple_filtering_config,
        )
        container = FilteringContainer()
        container.config.from_pydantic(config)

        container.wire(modules=[__name__], packages=["filtering"])

        assert isinstance(container.kg(), KG_CLASS_MAPPING[type(kg_config)])
        assert isinstance(
            container.node_filtering(), NODE_FILTERING_CLASS_MAPPING[type(node_filtering_config)]
        )
        assert isinstance(
            container.triple_filtering(),
            TRIPLE_FILTERING_CLASS_MAPPING[type(triple_filtering_config)],
        )
