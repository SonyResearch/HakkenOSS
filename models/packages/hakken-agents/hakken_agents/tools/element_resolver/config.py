from pydantic import BaseModel, Field

from hakken_agents.config import LLMConfig
from hakken_agents.config.embedder import EmbedderConfig
from hakken_agents.db.config import PostgresDBConfig
from hakken_agents.vector_db.config import VectorDBConfig, VectorDBTableConfig


class ElementResolverConfig(BaseModel):
    llm: LLMConfig = Field(description="The LLM to use for document resolution")
    db: PostgresDBConfig = Field(description="The database to use for document resolution")
    table: VectorDBTableConfig = Field(description="The table to use for document resolution")
    embedder: EmbedderConfig = Field(description="The embedder to use for document resolution")

    @property
    def vector_db(self) -> VectorDBConfig:
        return VectorDBConfig(
            db=self.db,
            embedder=self.embedder,
            table=self.table,
        )
