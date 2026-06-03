from typing import Any

import psycopg
from loguru import logger
from psycopg import rows

from hakken_agents.db.config import PostgresDBConfig, SQLTableConfig


class PostgresTable:
    """
    Lightweight table wrapper using config-driven approach.

    Example instantiation:
        config = PostgresDBConfig()
        table_cfg = SQLTableConfig(
            name="users",
            columns=[
                ColumnInfo(name="id",   dtype="BIGSERIAL PRIMARY KEY"),
                ColumnInfo(name="name", dtype="TEXT NOT NULL"),
                ColumnInfo(name="age",  dtype="INTEGER"),
            ]
        )
        table = PostgresTable(config, table_cfg)
    """

    def __init__(
        self,
        db_config: PostgresDBConfig,
        table_config: SQLTableConfig,
        *,
        autocommit: bool = False,
        min_pool_size: int = 4,
        max_pool_size: int = 20,
    ):
        self.db_config = db_config
        self.table_config = table_config
        self.table_name = table_config.qualified_name
        self.autocommit = autocommit

        # Recommended: use a real connection pool in production
        from psycopg_pool import ConnectionPool

        self.pool = ConnectionPool(
            conninfo=db_config.get_connection_string(),
            min_size=min_pool_size,
            max_size=max_pool_size,
            timeout=15,  # getconn timeout
            max_waiting=20,  # queue size
            max_lifetime=60 * 30,  # 30 min
            max_idle=5 * 60,  # 5 min idle → recycle
            reset=lambda conn: conn.rollback(),
            open=False,
        )
        self.pool.open()

        self._create_table_if_needed()

    def _create_table_if_needed(self):
        if not self.table_config.columns:
            logger.info("No columns defined → skipping automatic CREATE TABLE")
            return

        sql = self.table_config.get_create_table_sql()
        try:
            with self.pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql)
                conn.commit()
            logger.info(f"Ensured table exists: {self.table_name}")
        except psycopg.Error as e:
            logger.error(f"Failed to create/verify table {self.table_name}: {e}")
            raise

    def insert(self, data: dict[str, Any]) -> Any:
        if not data:
            raise ValueError("No data to insert")

        columns = list(data.keys())
        cols_str = ", ".join(columns)
        placeholders = ", ".join(f"%({k})s" for k in columns)

        sql = f"""
        INSERT INTO {self.table_name} ({cols_str})
        VALUES ({placeholders})
        RETURNING id;
        """

        with self.pool.connection() as conn:
            with conn.cursor(row_factory=rows.dict_row) as cur:
                cur.execute(sql, data)
                row = cur.fetchone()
                inserted_id = row["id"] if row else None

            if not self.autocommit:
                conn.commit()

        return inserted_id

    def insert_many(self, rows: list[dict[str, Any]]) -> None:
        """Insert multiple rows. All rows must have the same keys. No RETURNING."""
        if not rows:
            return
        columns = list(rows[0].keys())
        cols_str = ", ".join(columns)
        placeholders = ", ".join(f"%({k})s" for k in columns)
        sql = f"""
        INSERT INTO {self.table_name} ({cols_str})
        VALUES ({placeholders})
        """
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(sql, rows)
            if not self.autocommit:
                conn.commit()

    def read(
        self,
        where: str | None = None,
        params: dict[str, Any] | tuple | None = None,
        order_by: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        sql = f"SELECT * FROM {self.table_name}"
        if where:
            sql += f" WHERE {where}"
        if order_by:
            sql += f" ORDER BY {order_by}"
        if limit is not None:
            sql += f" LIMIT {limit}"

        with self.pool.connection() as conn:
            with conn.cursor(row_factory=rows.dict_row) as cur:
                cur.execute(sql, params or ())
                return cur.fetchall()

    def update(
        self,
        data: dict[str, Any],
        where: str,
        where_params: dict[str, Any] | None = None,
    ) -> int:
        """Update rows. WHERE clause must use named placeholders, e.g. \"id = %(id)s\"."""
        if not data:
            raise ValueError("No fields to update")

        sets = ", ".join(f"{k} = %({k})s" for k in data)
        sql = f"UPDATE {self.table_name} SET {sets} WHERE {where}"

        merged_params = {**data, **(where_params or {})}

        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, merged_params)
                rowcount = cur.rowcount

            if not self.autocommit:
                conn.commit()

        return rowcount

    def close(self):
        if hasattr(self, "pool") and self.pool:
            self.pool.close()
            logger.debug("Connection pool closed")

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
