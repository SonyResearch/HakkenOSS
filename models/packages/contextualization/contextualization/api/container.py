from dependency_injector import containers, providers
from pydantic_settings import BaseSettings, SettingsConfigDict

from contextualization.core.entities.config import (
    ContextSummarizerConfig,
    PublicationScorerConfig,
    ReferenceDatabaseConfig,
)
from contextualization.core.entities.config.retriever import RetrieverConfig
from contextualization.initialize import (
    initialize_context_summarizer,
    initialize_publication_scorer,
    initialize_reference_database,
    initialize_retriever,
)


class ApiConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_nested_delimiter="__", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    reference_database_config: ReferenceDatabaseConfig
    publication_scorer_config: PublicationScorerConfig
    context_summarizer_config: ContextSummarizerConfig | None = None
    retriever_config: RetrieverConfig


class ContextualizationContainer(containers.DeclarativeContainer):
    config = providers.Configuration()

    reference_database = providers.Singleton(
        initialize_reference_database, config=config.reference_database_config
    )
    publication_scorer = providers.Singleton(
        initialize_publication_scorer,
        config=config.publication_scorer_config,
        reference_database=reference_database,
    )
    context_summarizer = providers.Singleton(
        initialize_context_summarizer, config=config.context_summarizer_config
    )
    retriever = providers.Singleton(
        initialize_retriever,
        config=config.retriever_config,
        reference_database=reference_database,
        publication_scorer=publication_scorer,
        context_summarizer=context_summarizer,
    )
