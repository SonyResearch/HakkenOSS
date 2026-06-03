from dependency_injector import containers, providers
from pydantic_settings import BaseSettings, SettingsConfigDict
from spaice_inference_api.impl.logging.logging_impl import SpaiceLogger

from filtering import initialize
from filtering.core.entities.config.knowledge_graph import KnowledgeGraphConfig
from filtering.core.entities.config.node_filtering import (
    NodeFilteringConfig,
    RandomNodeFilteringConfig,
)
from filtering.core.entities.config.triple_filtering import (
    RandomTripleFilteringConfig,
    TripleFilteringConfig,
)


class FilteringSettings(BaseSettings):
    model_config = SettingsConfigDict(env_nested_delimiter="__")

    kg_config: KnowledgeGraphConfig
    node_filtering_config: NodeFilteringConfig = RandomNodeFilteringConfig()
    triple_filtering_config: TripleFilteringConfig = RandomTripleFilteringConfig()


class FilteringContainer(containers.DeclarativeContainer):
    config = providers.Configuration()

    logger = providers.Singleton(SpaiceLogger, __name__)

    kg = providers.Singleton(initialize.initialize_kg, config=config.kg_config)
    node_filtering = providers.Singleton(
        initialize.initialize_node_filtering, config=config.node_filtering_config, kg=kg
    )
    triple_filtering = providers.Singleton(
        initialize.initialize_triple_filtering, config=config.triple_filtering_config, kg=kg
    )
