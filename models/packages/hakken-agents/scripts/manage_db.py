"""
PostgreSQL Database Management CLI.

This script provides a Typer-based CLI for managing PostgreSQL databases:
- List tables
- Drop table(s)
- Describe table structure
- Query table data with filters

Usage:
    uv run python scripts/manage_db.py list-tables
    uv run python scripts/manage_db.py drop-table my_table
    uv run python scripts/manage_db.py drop-table 'agro_.*' --force --cascade
    uv run python scripts/manage_db.py drop-all-tables --schema public
    uv run python scripts/manage_db.py drop-all-tables --force --cascade
    uv run python scripts/manage_db.py describe-table my_table
    uv run python scripts/manage_db.py query-table my_table --limit 10
    uv run python scripts/manage_db.py query-table my_table --columns id name status --limit 5
    uv run python scripts/manage_db.py query-table my_table --filter status=active --limit 5
"""

import asyncio
import re
from typing import Annotated

import typer
from dotenv import load_dotenv
from loguru import logger

from hakken_agents.db.actions import PostgresActions
from hakken_agents.db.config import PostgresDBConfig

load_dotenv()

# ============================================================================
# Main App
# ============================================================================

app = typer.Typer(
    name="manage-db",
    help="PostgreSQL Database Management CLI",
    add_completion=False,
)


# ============================================================================
# Commands
# ============================================================================


@app.command("list-tables")
def list_tables(
    schema: Annotated[
        str,
        typer.Option("--schema", "-s", help="Schema to list tables from"),
    ] = "public",
) -> None:
    """
    List all tables in the PostgreSQL database with row counts.

    Shows table names and row count from the specified schema (default: public).
    """

    async def _run() -> None:
        actions = PostgresActions()
        try:
            tables = await actions.list_tables(schema=schema)

            if not tables:
                typer.echo(f"No tables found in schema '{schema}' ({actions.connection_info})")
                return

            typer.echo(f"📊 Tables in '{schema}' schema ({actions.connection_info}):")
            typer.echo("-" * 60)
            for table in tables:
                row_count = await actions.count_rows(table_name=table.name, schema=table.schema)
                typer.echo(f"  • {table.name}  ({row_count} rows)")
            typer.echo("-" * 60)
            typer.echo(f"Total: {len(tables)} table(s)")
        finally:
            await actions.close()

    try:
        asyncio.run(_run())
    except Exception as e:
        logger.error(f"Failed to list tables: {e}")
        typer.echo(f"❌ Error: {e}", err=True)
        raise typer.Exit(code=1)


@app.command("drop-table")
def drop_table(
    table_or_pattern: Annotated[
        str,
        typer.Argument(
            help="Table name or regex pattern matching table(s) to drop (e.g. my_table or agro_.*)",
        ),
    ],
    schema: Annotated[
        str,
        typer.Option("--schema", "-s", help="Schema containing the table(s)"),
    ] = "public",
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Skip confirmation prompt"),
    ] = False,
    cascade: Annotated[
        bool,
        typer.Option("--cascade", "-c", help="Drop dependent objects (CASCADE)"),
    ] = False,
) -> None:
    """
    Drop table(s) from the PostgreSQL database.

    The first argument is interpreted as a regex pattern: all tables in the schema
    whose name matches the pattern will be dropped. For a single table, use an
    exact pattern (e.g. ^my_table$) or the literal name (matches any table
    containing that substring). Example: agro_.* drops all tables whose names
    start with agro_.

    This operation is irreversible. Use --cascade to also drop dependent objects.
    """
    config = PostgresDBConfig()
    connection_info = f"{config.host}:{config.port}/{config.database}"

    try:
        pattern = re.compile(table_or_pattern)
    except re.error as e:
        typer.echo(f"❌ Invalid regex pattern: {e}", err=True)
        raise typer.Exit(code=1)

    async def _run() -> list[str]:
        actions = PostgresActions(config)
        try:
            tables = await actions.list_tables(schema=schema)
            matches = [t for t in tables if pattern.search(t.name)]
            return [t.name for t in matches]
        finally:
            await actions.close()

    try:
        matched_names = asyncio.run(_run())
    except Exception as e:
        logger.error(f"Failed to list tables: {e}")
        typer.echo(f"❌ Error: {e}", err=True)
        raise typer.Exit(code=1)

    if not matched_names:
        typer.echo(f"No tables in schema '{schema}' matched the pattern '{table_or_pattern}'.")
        raise typer.Exit(code=0)

    # Confirm deletion unless --force is used
    if not force:
        typer.echo(
            f"⚠️  Warning: The following table(s) in schema '{schema}' will be permanently "
            f"deleted from {connection_info}:"
        )
        for name in matched_names:
            typer.echo(f"   • {name}")
        if cascade:
            typer.echo("   CASCADE will also drop all dependent objects!")
        if not typer.confirm("Are you sure you want to continue?", default=False):
            typer.echo("Operation cancelled.")
            raise typer.Exit(code=0)

    async def _drop_all() -> None:
        actions = PostgresActions(config)
        try:
            for name in matched_names:
                await actions.drop_table(table_name=name, schema=schema, cascade=cascade)
                typer.echo(f"✅ Dropped {schema}.{name}")
        finally:
            await actions.close()

    try:
        asyncio.run(_drop_all())
    except Exception as e:
        logger.error(f"Failed to drop table(s): {e}")
        typer.echo(f"❌ Error: {e}", err=True)
        raise typer.Exit(code=1)


@app.command("drop-all-tables")
def drop_all_tables(
    schema: Annotated[
        str,
        typer.Option("--schema", "-s", help="Schema whose tables to drop"),
    ] = "public",
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Skip confirmation prompt"),
    ] = False,
    cascade: Annotated[
        bool,
        typer.Option("--cascade", "-c", help="Drop dependent objects (CASCADE)"),
    ] = False,
) -> None:
    """
    Drop all tables in a schema.

    This operation is irreversible and will permanently delete every table and its data.
    Use --cascade to also drop dependent objects.
    """

    async def _run() -> tuple[list[tuple[str, str]], str]:
        actions = PostgresActions()
        try:
            tables = await actions.list_tables(schema=schema)
            connection_info = actions.connection_info or ""
            return [(t.schema, t.name) for t in tables], connection_info
        finally:
            await actions.close()

    try:
        tables_to_drop, connection_info = asyncio.run(_run())
    except Exception as e:
        logger.error(f"Failed to list tables: {e}")
        typer.echo(f"❌ Error: {e}", err=True)
        raise typer.Exit(code=1)

    if not tables_to_drop:
        typer.echo(f"No tables found in schema '{schema}' ({connection_info}). Nothing to drop.")
        return

    if not force:
        typer.echo(
            f"⚠️  Warning: This will permanently delete all {len(tables_to_drop)} "
            f"table(s) in schema '{schema}' from {connection_info}"
        )
        for s, n in tables_to_drop:
            typer.echo(f"   • {s}.{n}")
        if cascade:
            typer.echo("   CASCADE will also drop all dependent objects!")
        if not typer.confirm("Are you sure you want to continue?", default=False):
            typer.echo("Operation cancelled.")
            raise typer.Exit(code=0)

    async def _drop_all() -> None:
        config = PostgresDBConfig()
        actions = PostgresActions(config)
        try:
            for table_schema, table_name in tables_to_drop:
                await actions.drop_table(
                    table_name=table_name, schema=table_schema, cascade=cascade
                )
                typer.echo(f"  Dropped {table_schema}.{table_name}")
            typer.echo(
                f"✅ Successfully dropped {len(tables_to_drop)} table(s) from {connection_info}"
            )
        finally:
            await actions.close()

    try:
        asyncio.run(_drop_all())
    except Exception as e:
        logger.error(f"Failed to drop tables: {e}")
        typer.echo(f"❌ Error: {e}", err=True)
        raise typer.Exit(code=1)


@app.command("describe-table")
def describe_table(
    table_name: Annotated[
        str,
        typer.Argument(help="Name of the table to describe"),
    ],
    schema: Annotated[
        str,
        typer.Option("--schema", "-s", help="Schema containing the table"),
    ] = "public",
) -> None:
    """
    Describe a table's structure (columns, types, constraints) and row count.
    """

    async def _run() -> None:
        actions = PostgresActions()
        try:
            columns = await actions.describe_table(table_name=table_name, schema=schema)

            if not columns:
                typer.echo(
                    f"❌ Table '{schema}.{table_name}' not found ({actions.connection_info})"
                )
                raise typer.Exit(code=1)

            row_count = await actions.count_rows(table_name=table_name, schema=schema)

            typer.echo(f"📋 Table '{schema}.{table_name}' ({actions.connection_info}):")
            typer.echo("-" * 80)
            typer.echo(f"{'Column':<30} {'Type':<20} {'Nullable':<10} {'Default':<20}")
            typer.echo("-" * 80)
            for col in columns:
                nullable = "YES" if col.nullable else "NO"
                default = str(col.default)[:18] if col.default else ""
                typer.echo(f"{col.name:<30} {col.type_display():<20} {nullable:<10} {default:<20}")
            typer.echo("-" * 80)
            typer.echo(f"Total: {len(columns)} column(s)  |  Rows: {row_count}")
        finally:
            await actions.close()

    try:
        asyncio.run(_run())
    except typer.Exit:
        raise
    except Exception as e:
        logger.error(f"Failed to describe table: {e}")
        typer.echo(f"❌ Error: {e}", err=True)
        raise typer.Exit(code=1)


def _parse_filter(value: str) -> tuple[str, str | int | float]:
    """Parse a single filter string 'col=value'. First '=' is the separator."""
    if "=" not in value:
        raise typer.BadParameter(f"Filter must be in form col=value, got: {value!r}")
    col, _, rest = value.partition("=")
    col = col.strip()
    val = rest.strip()
    if not col:
        raise typer.BadParameter(f"Filter column name is empty in: {value!r}")
    # Coerce value: int -> float -> str
    try:
        return (col, int(val))
    except ValueError:
        pass
    try:
        return (col, float(val))
    except ValueError:
        pass
    return (col, val)


@app.command("query-table")
def query_table(
    table_name: Annotated[
        str,
        typer.Argument(help="Name of the table to query"),
    ],
    schema: Annotated[
        str,
        typer.Option("--schema", "-s", help="Schema containing the table"),
    ] = "public",
    limit: Annotated[
        int | None,
        typer.Option("--limit", "-n", help="Maximum number of rows to return"),
    ] = None,
    offset: Annotated[
        int | None,
        typer.Option("--offset", help="Number of rows to skip"),
    ] = None,
    columns_arg: Annotated[
        list[str] | None,
        typer.Option(
            "--columns",
            "-c",
            help="Column names to select (repeatable). Omit to select all columns.",
        ),
    ] = None,
    filter_arg: Annotated[
        list[str] | None,
        typer.Option(
            "--filter",
            "-f",
            help="Equality filter as col=value (repeatable). Example: --filter status=active",
        ),
    ] = None,
) -> None:
    """
    Return data from a table with optional columns, filters, limit, and offset.

    Use --columns col1 col2 to select specific columns; omit for all columns.
    Use --filter col=value to restrict rows (equality; repeat for multiple columns).
    """

    filters: dict[str, str | int | float] = {}
    if filter_arg:
        for s in filter_arg:
            col, val = _parse_filter(s)
            filters[col] = val

    async def _run() -> None:
        actions = PostgresActions()
        try:
            rows = await actions.select_from_table(
                table_name=table_name,
                schema=schema,
                columns=columns_arg,
                limit=limit,
                offset=offset,
                filters=filters if filters else None,
            )

            if rows is None or (isinstance(rows, list) and len(rows) == 0):
                typer.echo(
                    f"No rows in '{schema}.{table_name}' "
                    f"(filters: {filters or 'none'}) ({actions.connection_info})"
                )
                return

            # Table header from first row keys
            keys = list(rows[0].keys())
            col_widths = [
                max([len(str(k))] + [len(str(r.get(k) or "")) for r in rows]) for k in keys
            ]
            col_widths = [min(w, 40) for w in col_widths]  # cap width for display

            typer.echo(f"📄 Table '{schema}.{table_name}' ({actions.connection_info}):")
            typer.echo("-" * (sum(col_widths) + 3 * (len(keys) - 1)))
            header = "  ".join(str(k)[:40].ljust(col_widths[i]) for i, k in enumerate(keys))
            typer.echo(header)
            typer.echo("-" * (sum(col_widths) + 3 * (len(keys) - 1)))
            for r in rows:
                cells = []
                for i, k in enumerate(keys):
                    v = r.get(k)
                    s = "" if v is None else str(v)
                    if len(s) > 40:
                        s = s[:37] + "..."
                    cells.append(s.ljust(col_widths[i]))
                typer.echo("  ".join(cells))
            typer.echo("-" * (sum(col_widths) + 3 * (len(keys) - 1)))
            typer.echo(f"Total: {len(rows)} row(s)")
        finally:
            await actions.close()

    try:
        asyncio.run(_run())
    except ValueError as e:
        logger.error(f"Invalid filter: {e}")
        typer.echo(f"❌ {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"Failed to query table: {e}")
        typer.echo(f"❌ Error: {e}", err=True)
        raise typer.Exit(code=1)


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    app()
