"""PostgreSQL database actions.

This module provides async functions for common PostgreSQL operations:
- Listing tables
- Dropping tables
- Describing table structure
- Row count
- Query table data with filters
"""

from dataclasses import dataclass
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from hakken_agents.db.config import PostgresDBConfig


@dataclass
class ColumnInfo:
    """Information about a database column."""

    name: str
    data_type: str
    nullable: bool
    default: str | None
    max_length: int | None

    def type_display(self) -> str:
        """Get display string for the column type."""
        if self.max_length:
            return f"{self.data_type}({self.max_length})"
        return self.data_type


@dataclass
class TableInfo:
    """Information about a database table."""

    schema: str
    name: str


class PostgresActions:
    """PostgreSQL database actions.

    Provides async methods for common database operations.
    Can be initialized with a config (creates its own engine) or with an existing engine.
    """

    def __init__(
        self,
        config: PostgresDBConfig | None = None,
        engine: AsyncEngine | None = None,
    ) -> None:
        """Initialize PostgreSQL actions.

        Args:
            config: PostgreSQL configuration. If None and no engine provided,
                loads from environment.
            engine: Optional existing AsyncEngine to use. If provided, the engine
                is not owned by this instance and won't be disposed on close().
        """
        self._engine: AsyncEngine | None = engine
        self._owns_engine = engine is None  # Only dispose if we created the engine
        self.config = (
            config if config is not None else (PostgresDBConfig() if engine is None else None)
        )

    @classmethod
    def from_engine(cls, engine: AsyncEngine) -> "PostgresActions":
        """Create PostgresActions from an existing AsyncEngine.

        Args:
            engine: An existing AsyncEngine instance.

        Returns:
            PostgresActions instance that uses the provided engine.
        """
        return cls(engine=engine)

    @property
    def connection_info(self) -> str | None:
        """Get connection info string for display."""
        if self.config is None:
            return None
        return f"{self.config.host}:{self.config.port}/{self.config.database}"

    async def _get_engine(self) -> AsyncEngine:
        """Get or create the async engine."""
        if self._engine is None:
            if self.config is None:
                raise RuntimeError("No engine or config provided to PostgresActions")
            connection_string = self.config.get_connection_string(is_async=True)
            self._engine = create_async_engine(connection_string)
        return self._engine

    async def close(self) -> None:
        """Close the database connection if owned by this instance."""
        if self._engine is not None and self._owns_engine:
            await self._engine.dispose()
            self._engine = None

    async def list_tables(self, schema: str = "public") -> list[TableInfo]:
        """List all tables in the specified schema.

        Args:
            schema: Schema to list tables from. Defaults to "public".

        Returns:
            List of TableInfo objects with schema and table names.
        """
        engine = await self._get_engine()
        query = text(
            """
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_schema = :schema
            AND table_type = 'BASE TABLE'
            ORDER BY table_name
        """
        )

        async with engine.connect() as conn:
            result = await conn.execute(query, {"schema": schema})
            return [TableInfo(schema=row[0], name=row[1]) for row in result]

    async def drop_table(
        self,
        table_name: str,
        schema: str = "public",
        cascade: bool = False,
    ) -> bool:
        """Drop a table from the database.

        Args:
            table_name: Name of the table to drop.
            schema: Schema containing the table. Defaults to "public".
            cascade: If True, drop dependent objects. Defaults to False.

        Returns:
            True if the operation completed (table may or may not have existed).
        """
        engine = await self._get_engine()
        cascade_clause = " CASCADE" if cascade else ""
        # Use proper identifier quoting to prevent SQL injection
        query = text(f'DROP TABLE IF EXISTS "{schema}"."{table_name}"{cascade_clause}')

        async with engine.connect() as conn:
            await conn.execute(query)
            await conn.commit()
        return True

    async def describe_table(
        self,
        table_name: str,
        schema: str = "public",
    ) -> list[ColumnInfo]:
        """Get detailed information about a table's columns.

        Args:
            table_name: Name of the table to describe.
            schema: Schema containing the table. Defaults to "public".

        Returns:
            List of ColumnInfo objects describing each column.
            Empty list if table doesn't exist.
        """
        engine = await self._get_engine()
        query = text(
            """
            SELECT
                column_name,
                data_type,
                is_nullable,
                column_default,
                character_maximum_length
            FROM information_schema.columns
            WHERE table_schema = :schema
            AND table_name = :table_name
            ORDER BY ordinal_position
        """
        )

        async with engine.connect() as conn:
            result = await conn.execute(query, {"schema": schema, "table_name": table_name})
            return [
                ColumnInfo(
                    name=row[0],
                    data_type=row[1],
                    nullable=row[2] == "YES",
                    default=row[3],
                    max_length=row[4],
                )
                for row in result
            ]

    async def count_rows(self, table_name: str, schema: str = "public") -> int:
        """Return the number of rows in a table.

        Args:
            table_name: Name of the table.
            schema: Schema containing the table. Defaults to "public".

        Returns:
            Row count. Returns 0 if the table does not exist or is empty.
        """
        engine = await self._get_engine()
        # Use proper identifier quoting to prevent SQL injection
        query = text(f'SELECT COUNT(*) FROM "{schema}"."{table_name}"')

        async with engine.connect() as conn:
            result = await conn.execute(query)
            row = result.fetchone()
            return int(row[0]) if row else 0

    async def select_from_table(
        self,
        table_name: str,
        schema: str = "public",
        *,
        columns: list[str] | None = None,
        limit: int | None = None,
        offset: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Select rows from a table with optional columns, filters, limit, and offset.

        Filter column names are validated against the table's columns.
        All filter values are passed as parameters to prevent SQL injection.

        Args:
            table_name: Name of the table.
            schema: Schema containing the table. Defaults to "public".
            columns: Optional list of column names to select. If None, selects all (*).
            limit: Maximum number of rows to return.
            offset: Number of rows to skip.
            filters: Optional dict of column name -> value (equality only).

        Returns:
            List of rows as dicts (column name -> value).
            Empty list if table does not exist.

        Raises:
            ValueError: If a filter key or requested column is not a valid column name.
        """
        table_columns = await self.describe_table(table_name=table_name, schema=schema)
        if not table_columns:
            return []
        valid_columns = {c.name for c in table_columns}

        if filters:
            invalid = set(filters) - valid_columns
            if invalid:
                raise ValueError(f"Invalid filter column(s): {invalid}")

        if columns is not None:
            invalid_cols = set(columns) - valid_columns
            if invalid_cols:
                raise ValueError(f"Invalid column(s): {invalid_cols}")
            select_list = ", ".join(f'"{c}"' for c in columns)
        else:
            select_list = "*"

        engine = await self._get_engine()
        base_sql = f'SELECT {select_list} FROM "{schema}"."{table_name}"'
        params: dict[str, Any] = {}
        conditions: list[str] = []
        for i, (col, val) in enumerate((filters or {}).items()):
            key = f"f{i}"
            params[key] = val
            # Column names from describe_table are valid; quote identifier
            conditions.append(f'"{col}" = :{key}')
        if conditions:
            base_sql += " WHERE " + " AND ".join(conditions)
        if limit is not None:
            base_sql += " LIMIT :limit"
            params["limit"] = limit
        if offset is not None:
            base_sql += " OFFSET :offset"
            params["offset"] = offset

        query = text(base_sql)
        async with engine.connect() as conn:
            result = await conn.execute(query, params)
            rows = result.mappings().fetchall()
            return [dict(r) for r in rows]

    async def table_exists(self, table_name: str, schema: str = "public") -> bool:
        """Check if a table exists.

        Args:
            table_name: Name of the table to check.
            schema: Schema containing the table. Defaults to "public".

        Returns:
            True if table exists, False otherwise.
        """
        engine = await self._get_engine()
        query = text(
            """
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = :schema
                AND table_name = :table_name
            )
        """
        )

        async with engine.connect() as conn:
            result = await conn.execute(query, {"schema": schema, "table_name": table_name})
            row = result.fetchone()
            return bool(row[0]) if row else False

    async def execute_query(self, query: str, params: dict[str, Any] | None = None) -> list[Any]:
        """Execute a raw SQL query.

        Args:
            query: SQL query string.
            params: Optional query parameters.

        Returns:
            List of result rows.
        """
        engine = await self._get_engine()
        async with engine.connect() as conn:
            result = await conn.execute(text(query), params or {})
            return list(result.fetchall())

    async def execute_and_commit(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[Any]:
        """Execute a raw SQL statement, commit the transaction, and return result rows.

        Use for INSERT/UPDATE/DELETE with RETURNING or any statement that must be
        committed. Returns the list of rows (e.g. from RETURNING).
        """
        engine = await self._get_engine()
        async with engine.connect() as conn:
            result = await conn.execute(text(query), params or {})
            rows = list(result.fetchall()) if result.returns_rows else []
            await conn.commit()
            return rows
