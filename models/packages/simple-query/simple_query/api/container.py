from dependency_injector import providers
from dependency_injector.containers import DeclarativeContainer
from pydantic_settings import BaseSettings, SettingsConfigDict
from query_common.initialize import create_parser
from query_common.values.types import ParserType
from spaice_inference_api.impl.logging.logging_impl import SpaiceLogger

from simple_query.api.initialize import (
    initialize_kg,
    initialize_link_predictor,
    initialize_querying,
)
from simple_query.kg.entities.configs import KnowledgeGraphConfig
from simple_query.link_predictor.entities.configs import (
    LinkPredictorConfig,
)
from simple_query.query.entities.configs import QueryingConfig


class ApiConfig(BaseSettings):
    """
    Configuration for API.

    Note that the types of `*_config` values are "discriminated union",
    which automatically finds a corresponding config class based on
    the value of a discriminator (`config_type` in this case) field.
    """

    model_config = SettingsConfigDict(
        env_nested_delimiter="__", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    kg_config: KnowledgeGraphConfig
    link_predictor_config: LinkPredictorConfig
    querying_config: QueryingConfig

    parser_type: ParserType = ParserType.LARK


class SimpleQueryingContainer(DeclarativeContainer):
    config = providers.Configuration()

    logger = providers.Singleton(SpaiceLogger, __name__)

    parser = providers.Singleton(create_parser, parser_type=config.parser_type)
    kg = providers.Singleton(initialize_kg, config=config.kg_config)
    link_predictor = providers.Singleton(
        initialize_link_predictor, config=config.link_predictor_config
    )
    querying = providers.Singleton(
        initialize_querying, config=config.querying_config, kg=kg, link_predictor=link_predictor
    )
