from langchain_postgres.v2.engine import Column
from pydantic import BaseModel, Field

from hakken_agents.config.embedder import EmbedderConfig
from hakken_agents.db.config import PostgresDBConfig


class VectorDBTableConfig(BaseModel):
    name: str = Field(description="The name of the table")
    schema_name: str = Field(default="public", description="The name of the schema")
    content_column: str = Field(
        default="content", description="The name of the column to store the content"
    )
    embedding_column: str = Field(
        default="embedding", description="The name of the column to store the embedding"
    )
    metadata_columns: list[Column] | None = Field(
        default=None, description="The metadata columns to create in the table"
    )

    model_config = {"arbitrary_types_allowed": True}


class VectorDBConfig(BaseModel):
    db: PostgresDBConfig = Field(description="The database configuration")
    embedder: EmbedderConfig = Field(description="The embedder configuration")
    table: VectorDBTableConfig | None = Field(default=None, description="The table configuration")
