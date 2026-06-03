from __future__ import annotations

from pydantic import AliasChoices, BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ColumnInfo(BaseModel):
    """Description of a single column."""

    name: str = Field(description="Column name")
    dtype: str = Field(
        description="SQL type (e.g. 'BIGSERIAL', 'TEXT NOT NULL', 'UUID PRIMARY KEY', 'JSONB')"
    )
    # Future extensions (optional for now)
    # nullable: bool = Field(default=True)
    # default: Any | None = Field(default=None)
    # unique: bool = Field(default=False)


class SQLTableConfig(BaseModel):
    """Declarative configuration for one PostgreSQL table."""

    name: str = Field(default="entities", description="Table name")
    schema_name: str = Field(default="public", description="Schema name (usually 'public')")
    columns: list[ColumnInfo] = Field(
        default_factory=list,
        description="List of column definitions — used to generate CREATE TABLE",
    )
    constraints: list[str] = Field(
        default_factory=list,
        description="Table-level constraint clauses (e.g. 'UNIQUE (a, b)', 'PRIMARY KEY (id)')",
    )

    @property
    def qualified_name(self) -> str:
        return f"{self.schema_name}.{self.name}"

    def get_create_table_sql(self) -> str:
        if not self.columns:
            raise ValueError("Cannot generate CREATE TABLE: columns list is empty")

        cols = []
        for col in self.columns:
            cols.append(f"{col.name} {col.dtype}")
        parts = cols + self.constraints
        parts_str = ",\n    ".join(parts)
        return f"""
CREATE TABLE IF NOT EXISTS {self.qualified_name} (
    {parts_str}
);
        """.strip()


class PostgresDBConfig(BaseSettings):
    """PostgreSQL connection settings — loaded from environment / .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="POSTGRES_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    user: str
    password: str
    host: str = "localhost"
    port: int = 5432
    database: str = Field(
        validation_alias=AliasChoices("POSTGRES_DB", "POSTGRES_DATABASE", "db", "database")
    )

    def get_connection_string(self, is_async: bool = False) -> str:
        """Plain psycopg connection string (recommended for psycopg 3)."""
        if is_async:
            return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
