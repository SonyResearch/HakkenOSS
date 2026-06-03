from typing import Annotated, Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from data_api.core.values.types import DatabaseType


class APIConfig(BaseSettings):
    kv_store_url: str = "localhost"
    kv_store_port: int = 6379
    kv_store_db: int = 0


class DatabaseConfigBase(BaseModel):
    config_type: DatabaseType


class PostgresDatabaseConfig(DatabaseConfigBase):
    config_type: Literal[DatabaseType.POSTGRES] = DatabaseType.POSTGRES
    host: str = Field(description="hosr string for the Postgres database")
    port: str = Field(description="Port number for the Postgres database")
    db: str = Field(description="Database name for the Postgres database")
    user: str = Field(description="Username for the Postgres database")
    password: str = Field(description="Password for the Postgres database")


DatabaseConfig = Annotated[
    PostgresDatabaseConfig,
    Field(discriminator="config_type"),
]
