# Hakken Data Processing

A package that implements the cleaning and processing pipeline for the relations data to use in the Hakken System.


## 🚀 Features
- Data adapters for Pandas and PySpark
- Configurable processing pipeline
- Temporal KG Engine 
- CLI scripts for ingest and query
- MkDocs documentation (Material + mkdocstrings)


## 🛠️ Installation

Create env and install

```bash
uv venv
uv sync
```


## 📚 Documentation

The docs focus on the Temporal KG Engine operations and usage.

Serve locally:
```bash
mkdocs serve
```

Build:
```bash
mkdocs build
```

See the docs for complete setup, restore, verification, and operations.

## 🧾 Makefile

Common tasks are available via `make`:

- checks: Formats, lints, tests, and type-checks
- ingest-temporal-kg: Runs the temporal KG ingest script
- query-temporal-kg: Runs the temporal KG query script

Examples:

```bash
make checks
make ingest-temporal-kg
make query-temporal-kg
```



