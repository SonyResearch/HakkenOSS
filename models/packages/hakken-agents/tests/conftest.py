# tests/conftest.py
import pytest
from testcontainers.postgres import PostgresContainer

from hakken_agents.db.config import (  # adjust import path
    ColumnInfo,
    PostgresDBConfig,
    SQLTableConfig,
)
from hakken_agents.db.engine import PostgresTable


@pytest.fixture(scope="session")
def test_postgres_container() -> PostgresContainer:
    """
    Starts one PostgreSQL container for the entire test session.
    Provides connection info.
    """
    container = PostgresContainer(
        image="postgres:16-alpine",  # or "postgres:17-alpine" if available
        dbname="testdb",
        username="testuser",
        password="testpass",
    )
    container.start()
    # Optional: give extra time if startup is slow in CI
    # container.wait_ready(timeout=30)

    yield container

    container.stop()


@pytest.fixture(scope="session")
def test_db_config(test_postgres_container: PostgresContainer) -> PostgresDBConfig:
    """
    Fake PostgresDBConfig using the test container's credentials.
    This mimics how you'd load real config in production.
    """

    return PostgresDBConfig(
        user=test_postgres_container.username,
        password=test_postgres_container.password,
        host=test_postgres_container.get_container_host_ip(),
        port=test_postgres_container.get_exposed_port(5432),
        database=test_postgres_container.dbname,
    )


@pytest.fixture(scope="module")
def test_table_config() -> SQLTableConfig:
    """
    Example table config — override / parametrize in specific test files if needed.
    You can also make this a function-scoped fixture or pass different configs per test.
    """
    return SQLTableConfig(
        name="test_users",
        schema_name="public",
        columns=[
            ColumnInfo(name="id", dtype="BIGSERIAL PRIMARY KEY"),
            ColumnInfo(name="name", dtype="TEXT NOT NULL"),
            ColumnInfo(name="age", dtype="INTEGER"),
            ColumnInfo(name="email", dtype="TEXT UNIQUE"),
            ColumnInfo(name="created_at", dtype="TIMESTAMPTZ DEFAULT NOW()"),
        ],
    )


@pytest.fixture(scope="module")
def postgres_table(
    test_db_config: PostgresDBConfig,
    test_table_config: SQLTableConfig,
) -> PostgresTable:
    """
    Creates PostgresTable instance with real pool connected to test container.
    Table is auto-created via your _create_table_if_needed().
    """
    table = PostgresTable(
        db_config=test_db_config,
        table_config=test_table_config,
        min_pool_size=2,  # smaller for tests
        max_pool_size=8,
    )
    yield table
    table.close()  # ensures pool is closed cleanly
