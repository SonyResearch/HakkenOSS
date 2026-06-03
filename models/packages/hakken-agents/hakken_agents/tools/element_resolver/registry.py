"""Table registry — binds vector tables to their embedder config and schema.

Stores a row per table in ``element_resolver_registry`` so that the API can
discover the correct embedder model/dim and metadata columns at startup,
and the ingest CLI can detect embedder mismatches before writing data.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime

from langchain_postgres.v2.engine import Column
from loguru import logger
from pydantic import BaseModel, Field

from hakken_agents.config.embedder import EmbedderConfig
from hakken_agents.db.actions.postgres import PostgresActions
from hakken_agents.db.config import PostgresDBConfig
from hakken_agents.vector_db.config import VectorDBTableConfig

REGISTRY_TABLE = "element_resolver_registry"


class TableRegistryEntry(BaseModel):
    """A registry row binding a vector table to its embedder and schema."""

    table_name: str
    embedder_model: str
    embedder_dim: int
    embedder_base_url: str | None = None
    metadata_columns: list[dict[str, str]] = Field(default_factory=list)
    created_at: datetime | None = None

    @classmethod
    def from_config(
        cls,
        table_config: VectorDBTableConfig,
        embedder_config: EmbedderConfig,
    ) -> TableRegistryEntry:
        """Build an entry from existing config objects."""
        metadata_cols: list[dict[str, str]] = []
        if table_config.metadata_columns:
            metadata_cols = [
                {"name": col.name, "data_type": col.data_type}
                for col in table_config.metadata_columns
            ]
        return cls(
            table_name=table_config.name,
            embedder_model=embedder_config.embedding_model,
            embedder_dim=embedder_config.embedding_dim,
            embedder_base_url=embedder_config.base_url,
            metadata_columns=metadata_cols,
        )

    def to_table_config(self) -> VectorDBTableConfig:
        """Reconstruct a VectorDBTableConfig from this entry."""
        metadata_columns = (
            [
                Column(name=col["name"], data_type=col.get("data_type", "TEXT"))
                for col in self.metadata_columns
            ]
            if self.metadata_columns
            else None
        )
        return VectorDBTableConfig(
            name=self.table_name,
            metadata_columns=metadata_columns,
        )


class TableRegistry:
    """CRUD operations for the ``element_resolver_registry`` table.

    Each method opens and closes its own DB connection so that the registry
    is safe to use in both sync (CLI via ``asyncio.run``) and async (API
    lifespan) contexts without event-loop conflicts.
    """

    def __init__(self, db_config: PostgresDBConfig) -> None:
        self._db_config = db_config

    @asynccontextmanager
    async def _connect(self) -> AsyncIterator[PostgresActions]:
        db = PostgresActions(config=self._db_config)
        try:
            yield db
        finally:
            await db.close()

    @staticmethod
    async def _ensure_table(db: PostgresActions) -> None:
        await db.execute_and_commit(
            f"""
            CREATE TABLE IF NOT EXISTS public."{REGISTRY_TABLE}" (
                table_name  TEXT PRIMARY KEY,
                embedder_model TEXT NOT NULL,
                embedder_dim   INTEGER NOT NULL,
                embedder_base_url TEXT,
                metadata_columns JSONB NOT NULL DEFAULT '[]'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        await db.execute_and_commit(
            f"""
            ALTER TABLE public."{REGISTRY_TABLE}"
            ADD COLUMN IF NOT EXISTS embedder_base_url TEXT
            """
        )

    async def get(self, table_name: str) -> TableRegistryEntry | None:
        """Read a registry entry. Returns ``None`` if not found."""
        async with self._connect() as db:
            await self._ensure_table(db)
            rows = await db.execute_query(
                f"""
                SELECT table_name, embedder_model, embedder_dim,
                       embedder_base_url, metadata_columns, created_at
                FROM public."{REGISTRY_TABLE}"
                WHERE table_name = :table_name
                """,
                {"table_name": table_name},
            )
        if not rows:
            return None
        row = rows[0]
        raw_cols = row[4]
        metadata_columns = raw_cols if isinstance(raw_cols, list) else json.loads(raw_cols)
        return TableRegistryEntry(
            table_name=row[0],
            embedder_model=row[1],
            embedder_dim=row[2],
            embedder_base_url=row[3],
            metadata_columns=metadata_columns,
            created_at=row[5],
        )

    async def upsert(self, entry: TableRegistryEntry) -> None:
        """Insert or update a registry entry."""
        cols_json = json.dumps(entry.metadata_columns)
        async with self._connect() as db:
            await self._ensure_table(db)
            await db.execute_and_commit(
                f"""
                INSERT INTO public."{REGISTRY_TABLE}"
                    (table_name, embedder_model, embedder_dim, embedder_base_url, metadata_columns)
                VALUES
                    (:table_name, :embedder_model, :embedder_dim, :embedder_base_url,
                     CAST(:metadata_columns AS jsonb))
                ON CONFLICT (table_name) DO UPDATE SET
                    embedder_model    = EXCLUDED.embedder_model,
                    embedder_dim      = EXCLUDED.embedder_dim,
                    embedder_base_url = EXCLUDED.embedder_base_url,
                    metadata_columns  = EXCLUDED.metadata_columns
                """,
                {
                    "table_name": entry.table_name,
                    "embedder_model": entry.embedder_model,
                    "embedder_dim": entry.embedder_dim,
                    "embedder_base_url": entry.embedder_base_url,
                    "metadata_columns": cols_json,
                },
            )
        logger.info(f"Registry: upserted entry for table '{entry.table_name}'")

    async def validate_or_register(self, entry: TableRegistryEntry) -> TableRegistryEntry:
        """Validate an existing entry or register a new one.

        If an entry already exists for the table, validates that the embedder
        model and dimension match.  Raises ``ValueError`` on mismatch.
        If no entry exists, creates one.
        """
        existing = await self.get(entry.table_name)

        if existing is not None:
            if existing.embedder_model != entry.embedder_model:
                raise ValueError(
                    f"Embedder model mismatch for table '{entry.table_name}': "
                    f"registry has '{existing.embedder_model}', "
                    f"got '{entry.embedder_model}'. "
                    f"Cannot change the embedder for an existing table."
                )
            if existing.embedder_dim != entry.embedder_dim:
                raise ValueError(
                    f"Embedder dimension mismatch for table '{entry.table_name}': "
                    f"registry has {existing.embedder_dim}, got {entry.embedder_dim}."
                )
            logger.info(f"Registry: validated entry for table '{entry.table_name}'")
            return existing

        await self.upsert(entry)
        return entry

    def get_sync(self, table_name: str) -> TableRegistryEntry | None:
        return asyncio.run(self.get(table_name))

    def validate_or_register_sync(self, entry: TableRegistryEntry) -> TableRegistryEntry:
        return asyncio.run(self.validate_or_register(entry))
