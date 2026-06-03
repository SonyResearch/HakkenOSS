from typing import Self

from dependency_injector import providers
from dependency_injector.containers import DeclarativeContainer
from filtering.container import FilteringContainer, FilteringSettings
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from query_common.initialize import create_parser
from query_common.values.types import ParserType
from spaice_inference_api.impl.logging.logging_impl import SpaiceLogger

from complex_query import initialize
from complex_query.core.entities.config.kg import KGConfig, Neo4jKGConfig
from complex_query.core.entities.config.kg_ledger import HDF5KGLedgerConfig, KGLedgerConfig
from complex_query.core.entities.config.link_predictor import (
    ApiBasedLinkPredictorConfig,
    LinkPredictorConfig,
)
from complex_query.core.entities.config.score_aggregator import (
    ProductScoreAggregatorConfig,
    ScoreAggregatorConfig,
)
from complex_query.core.entities.config.score_ledger import (
    InMemoryScoreLedgerConfig,
    ScoreLedgerConfig,
)
from complex_query.core.entities.config.search import BeamSearchConfig, SearchConfig


class QueryingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_nested_delimiter="__", env_file=".env", extra="ignore")

    parser_type: ParserType = ParserType.LARK

    search_config: SearchConfig = BeamSearchConfig()
    kg_config: KGConfig = Neo4jKGConfig()
    score_ledger_config: ScoreLedgerConfig = InMemoryScoreLedgerConfig()
    link_predictor_config: LinkPredictorConfig = ApiBasedLinkPredictorConfig()
    score_aggregator_config: ScoreAggregatorConfig = ProductScoreAggregatorConfig()
    kg_ledger_config: KGLedgerConfig = HDF5KGLedgerConfig()

    use_kg_ledger: bool = True
    use_filtering: bool = True

    filtering_container_config: FilteringSettings | None = None

    @model_validator(mode="after")
    def check_kg_ledger_values(self) -> "Self":
        if self.use_kg_ledger and not self.kg_ledger_config:
            raise ValueError("kg_ledger_config must be given when use_kg_ledger is True.")
        return self

    @model_validator(mode="after")
    def check_filtering_container_config_values(self) -> "Self":
        if self.use_filtering and not self.filtering_container_config:
            raise ValueError("filtering_container_config must be given when use_filtering is True.")
        return self


class QueryingContainer(DeclarativeContainer):
    config = providers.Configuration()

    logger = providers.Singleton(SpaiceLogger, __name__)

    filtering_container = providers.Container(
        FilteringContainer, config=config.filtering_container_config
    )
    node_filtering = providers.Selector(
        config.use_filtering.as_(lambda true_or_false: str(true_or_false).lower()),
        true=filtering_container.node_filtering,
        false=providers.Object(None),
    )

    parser = providers.Singleton(create_parser, parser_type=config.parser_type)
    kg = providers.Singleton(
        initialize.initialize_kg,
        config=config.kg_config,
        use_kg_ledger=config.use_kg_ledger,
        kg_ledger_config=config.kg_ledger_config,
    )
    score_ledger = providers.Singleton(
        initialize.initialize_score_ledger, config=config.score_ledger_config
    )
    score_aggregator = providers.Singleton(
        initialize.initialize_score_aggregator, config=config.score_aggregator_config
    )
    link_predictor = providers.Singleton(
        initialize.initialize_link_predictor, config=config.link_predictor_config
    )
    search = providers.Singleton(
        initialize.initialize_search,
        config=config.search_config,
        kg=kg,
        score_ledger=score_ledger,
        link_predictor=link_predictor,
        node_filtering=node_filtering,
    )
