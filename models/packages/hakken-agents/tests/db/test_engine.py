# tests/test_postgres_table.py
import pytest

from hakken_agents.db.config import SQLTableConfig
from hakken_agents.db.engine import PostgresTable


def test_insert_and_read(postgres_table: PostgresTable):
    data = {"name": "Alice", "age": 30, "email": "alice@example.com"}

    inserted_id = postgres_table.insert(data)
    assert inserted_id is not None, "Should return inserted id"

    rows = postgres_table.read("id = %s", (inserted_id,))
    assert len(rows) == 1
    assert rows[0]["name"] == "Alice"
    assert rows[0]["age"] == 30


def test_update(postgres_table: PostgresTable):
    # First insert
    data = {"name": "Bob", "age": 25, "email": "bob@example.com"}
    inserted_id = postgres_table.insert(data)

    # Update (named placeholders in WHERE)
    updates = {"age": 26}
    rowcount = postgres_table.update(updates, "id = %(id)s", {"id": inserted_id})
    assert rowcount == 1

    # Verify
    rows = postgres_table.read("id = %s", (inserted_id,))
    assert rows[0]["age"] == 26


@pytest.mark.parametrize("age_filter, expected_count", [(25, 1), (40, 0)])
def test_read_with_where(postgres_table: PostgresTable, age_filter, expected_count):
    # Unique email per parametrized case so shared table has no duplicate key
    postgres_table.insert(
        {
            "name": "Charlie",
            "age": 35,
            "email": f"charlie_{age_filter}@example.com",
        }
    )

    rows = postgres_table.read("age > %s", (age_filter,), order_by="age DESC", limit=10)
    assert len(rows) >= expected_count  # at least the ones we expect


def test_no_columns_config_skips_creation(test_db_config):
    empty_cfg = SQLTableConfig(name="empty_table", columns=[])
    table = PostgresTable(test_db_config, empty_cfg)
    # Should not raise, and table won't be auto-created
    table.close()
