from typing import cast

from omegaconf import DictConfig, OmegaConf
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from hakken_agents.config import DocumentConfig, EmbedderConfig
from hakken_agents.config.llm import LLMConfig
from hakken_agents.db.config import PostgresDBConfig
from hakken_agents.enki.nodes.domain_resolver import DomainResolverConfig
from hakken_agents.enki.nodes.entity_extractor import EntityExtractorConfig
from hakken_agents.enki.nodes.entity_resolver import EntityResolverConfig
from hakken_agents.enki.nodes.fact_extractor import FactExtractorConfig
from hakken_agents.enki.nodes.fact_resolver import FactResolverConfig
from hakken_agents.tools.element_resolver import ElementResolverConfig
from hakken_agents.vector_db.config import VectorDBTableConfig


class EnkiConfig(BaseSettings):
    """Configuration for the Graph Builder."""

    model_config = SettingsConfigDict(extra="ignore")

    db: PostgresDBConfig = Field(default_factory=PostgresDBConfig)
    document: DocumentConfig = Field(default_factory=DocumentConfig)
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    entity_extractor: EntityExtractorConfig = Field(default_factory=EntityExtractorConfig)
    fact_extractor: FactExtractorConfig = Field(default_factory=FactExtractorConfig)
    chunks_table: VectorDBTableConfig = Field(default_factory=VectorDBTableConfig)
    entities_table: VectorDBTableConfig = Field(default_factory=VectorDBTableConfig)
    domains_table: VectorDBTableConfig = Field(default_factory=VectorDBTableConfig)
    relations_table: VectorDBTableConfig = Field(default_factory=VectorDBTableConfig)
    facts_table_name: str = Field(
        default="facts",
        description="Name of the relational facts table.",
    )
    workspace: str = Field(
        default="enki-ingest",
        description="Workspace id for pipeline/cache isolation.",
    )

    @property
    def chunk_resolver(self) -> ElementResolverConfig:
        return ElementResolverConfig(
            llm=self.llm,
            db=self.db,
            embedder=self.embedder,
            table=self.chunks_table,
            content_fields=["text"],
            context_fields=["heading"],
        )

    @property
    def domain_resolver(self) -> DomainResolverConfig:
        return DomainResolverConfig(
            llm=self.llm,
            db=self.db,
            embedder=self.embedder,
            table=self.domains_table,
            content_fields=["name"],
            context_fields=[],
        )

    @property
    def entity_resolver(self) -> EntityResolverConfig:
        return EntityResolverConfig(
            llm=self.llm,
            db=self.db,
            embedder=self.embedder,
            table=self.entities_table,
            content_fields=["name", "description"],
            context_fields=[],
            threshold=0.95,
        )

    @property
    def fact_resolver(self) -> FactResolverConfig:
        return FactResolverConfig(
            relation_resolver=ElementResolverConfig(
                llm=self.llm,
                db=self.db,
                embedder=self.embedder,
                table=self.relations_table,
                content_fields=["name"],
                context_fields=[],
            ),
            db=self.db,
            facts_table=self.facts_table_name,
        )

    @classmethod
    def from_omegaconf(cls, cfg: DictConfig) -> "EnkiConfig":
        """Create BuildGraphConfig from OmegaConf dictConfig."""
        cfg_dict = cast(dict, OmegaConf.to_container(cfg, resolve=True))
        return cls(**cfg_dict)
