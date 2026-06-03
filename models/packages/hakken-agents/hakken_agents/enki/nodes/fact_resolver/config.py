from pydantic import BaseModel, Field

from hakken_agents.db.config import PostgresDBConfig
from hakken_agents.tools.element_resolver.config import ElementResolverConfig


class FactResolverConfig(BaseModel):
    """Config for fact resolution.

    Combines an ElementResolverConfig for relation deduplication (vector store)
    with a PostgresDBConfig + table name for relational fact storage via
    ``FactsTable``.
    """

    relation_resolver: ElementResolverConfig = Field(
        description="Config for the internal relation resolver (vector store)",
    )
    db: PostgresDBConfig = Field(
        description="Database connection for the facts relational table",
    )
    facts_table: str = Field(
        default="facts",
        description="Name of the facts relational table",
    )
