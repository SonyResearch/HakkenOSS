"""
Element Resolver CLI: ingest entities from TSV and query by similarity.

Usage:
    uv run python scripts/run_element_resolver.py ingest --data-uri s3://bucket/path/nodes.tsv --table-name my_table
    uv run python scripts/run_element_resolver.py ingest -i s3://bucket/path/nodes.tsv -t my_table -m openai/gpt-4o-mini
    uv run python scripts/run_element_resolver.py ingest -i file.tsv -t my_table --content-columns name,context --content-template "{{ name }}"
    uv run python scripts/run_element_resolver.py ingest -i file.tsv -t my_table --content-columns name --content-template "{{ name }}" --no-description
    uv run python scripts/run_element_resolver.py query "bacterium Desulfurivibrio" -t my_table --k 5
    uv run python scripts/run_element_resolver.py query "gene" -t my_table --filter '{"context":...}' -j
"""

import json
import math
import os
from pathlib import Path
from typing import Any

import polars as pl
import typer
from dotenv import load_dotenv
from jinja2 import Template
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache
from langchain_postgres.v2.engine import Column
from loguru import logger
from pydantic import SecretStr
from tqdm import tqdm

from hakken_agents.config import EmbedderConfig, LLMConfig
from hakken_agents.db.config import PostgresDBConfig
from hakken_agents.tools.element_resolver import (
    ElementResolver,
    ElementResolverConfig,
    TableRegistry,
    TableRegistryEntry,
)
from hakken_agents.tools.element_resolver.schemas import SimilaritySearchParam
from hakken_agents.vector_db.config import VectorDBTableConfig

load_dotenv()


_script_dir = Path(__file__).resolve().parent
_package_root = _script_dir.parent
_cache_dir = _package_root / "cache"
_cache_dir.mkdir(parents=True, exist_ok=True)
set_llm_cache(SQLiteCache(str(_cache_dir / "llm_cache.db")))

DATA_URI_TEMPLATE = "s3://sai-spaice-ds/data/processed/data_processing/{dataset_name}/{filename}"

app = typer.Typer(
    name="element-resolver",
    help="Ingest entities from TSV and query by similarity",
    add_completion=False,
)


@app.command("ingest")
def ingest(
    data_uri: str = typer.Option(
        ...,
        "--data-uri",
        "-i",
        help="Full S3 or local URI",
    ),
    table_name: str = typer.Option(
        ...,
        "--table-name",
        "-t",
        help="Table name for the vector store",
    ),
    content_columns_str: str = typer.Option(
        ...,
        "--content-columns",
        help="Comma-separated columns for the content field",
    ),
    content_template: str = typer.Option(
        ...,
        "--content-template",
        help="Template for the content field",
    ),
    limit: int | None = typer.Option(None, "--limit", "-n", help="Max rows to ingest"),
    model_name: str = typer.Option(
        "openai/gpt-4.1-nano",
        "--model",
        "-m",
        help="LLM model name for description generation",
    ),
    model_temperature: float = typer.Option(
        0.2,
        "--model-temperature",
        help="LLM temperature",
    ),
    model_base_url: str | None = typer.Option(
        None,
        "--model-base-url",
        help="LLM base URL (default: OPENROUTER_BASE_URL env)",
    ),
    embedder_name: str = typer.Option(
        "openai/text-embedding-3-small",
        "--embedder",
        "-e",
        help="Embedding model name",
    ),
    embedder_base_url: str | None = typer.Option(
        None,
        "--embedder-base-url",
        help="Embedder base URL (default: OPENROUTER_BASE_URL env)",
    ),
    embedding_dim: int = typer.Option(
        1024,
        "--embedding-dim",
        "-d",
        help="Embedding dimension (must match the embedder model)",
    ),
    batch_size: int = typer.Option(
        50,
        "--batch-size",
        "-b",
        help="Number of elements per LLM/insert batch",
    ),
    max_concurrency: int = typer.Option(
        10,
        "--max-concurrency",
        "-c",
        help="Max concurrent LLM calls within each batch",
    ),
    no_description: bool = typer.Option(
        False,
        "--no-description",
        help="Skip LLM descriptions; embed and store only the content.",
    ),
    max_description_tokens: int | None = typer.Option(
        100,
        "--max-description-tokens",
        help="Max tokens for each LLM-generated description (default 100). Use 0 for no limit.",
    ),
    unique: bool = typer.Option(
        False,
        "--unique",
        "-u",
        help="Deduplicate rows by content columns before ingesting.",
    ),
) -> None:
    """Ingest entities from a TSV file (S3 or local) into the vector store."""

    logger.info(f"Reading data from {data_uri}")

    df = pl.scan_csv(data_uri, separator="\t").collect()
    if unique:
        before = df.height
        df = df.unique(subset=content_columns_str.split(","))
        logger.info(f"Deduplicated: {before} → {df.height} rows (by {content_columns_str})")
    if limit is not None:
        df = df.head(limit)

    api_key = os.getenv("OPENROUTER_API_KEY", "")

    if "localhost" in model_base_url:
        logger.info("Using local model")
        model_api_key = ""
    else:
        model_api_key = api_key

    if "localhost" in embedder_base_url:
        logger.info("Using local embedder")
        embedder_api_key = ""
    else:
        embedder_api_key = api_key

    content_columns = content_columns_str.split(",")

    metadata_columns = [Column(name=c, data_type="TEXT") for c in content_columns]

    resolver_config = ElementResolverConfig(
        llm=LLMConfig(
            name=model_name,
            temperature=model_temperature,
            api_key=SecretStr(model_api_key),
            base_url=model_base_url,
            max_tokens=max_description_tokens or None,
        ),
        db=PostgresDBConfig(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", "postgres"),
            database=os.getenv("POSTGRES_DATABASE", "hakken_agents"),
        ),
        table=VectorDBTableConfig(
            name=table_name,
            schema_name="public",
            content_column="content",
            embedding_column="embedding",
            metadata_columns=metadata_columns,
        ),
        embedder=EmbedderConfig(
            embedding_model=embedder_name,
            embedding_dim=embedding_dim,
            api_key=SecretStr(embedder_api_key),
            base_url=embedder_base_url,
        ),
        content_fields=["name"],
        context_fields=["context"],
    )
    registry = TableRegistry(resolver_config.db)
    registry_entry = TableRegistryEntry.from_config(resolver_config.table, resolver_config.embedder)
    registry.validate_or_register_sync(registry_entry)

    resolver = ElementResolver.from_config(resolver_config)

    content_tpl: Template = Template(content_template)

    elements: list[dict[str, Any]] = []
    for row in df.iter_rows(named=True):
        row_str: dict[str, str] = {k: str(v).strip() for k, v in row.items()}
        content = content_tpl.render(**row_str)
        if not content.strip():
            continue
        metadata = {c: row_str.get(c, "") for c in content_columns}
        elements.append({"content": content, **metadata})

    total = len(elements)
    num_batches = math.ceil(total / batch_size)
    ingested = 0
    skipped_total = 0
    for i in tqdm(range(0, total, batch_size), total=num_batches, desc="Ingesting"):
        batch = elements[i : i + batch_size]
        documents = [resolver.to_document(**element) for element in batch]

        new_docs = [doc for doc in documents if not resolver.exists(doc.id)]
        skipped = len(documents) - len(new_docs)
        skipped_total += skipped
        if skipped:
            logger.info(f"Skipped {skipped} already-existing element(s)")
        if not new_docs:
            continue

        if not no_description:
            new_docs = resolver.add_descriptions_batch(new_docs, max_concurrency=max_concurrency)

        resolver.add_many(new_docs)
        ingested += len(new_docs)

    typer.echo(
        f"Done. Ingested {ingested}, skipped {skipped_total} existing element(s) "
        f"from {len(df)} row(s)."
    )


@app.command("query")
def query(
    query_text: str = typer.Argument(..., help="Search query text"),
    k: int = typer.Option(5, "--k", "-k", help="Number of results to return"),
    filter_json: str | None = typer.Option(
        None,
        "--filter",
        "-f",
        help='Metadata filter as JSON, e.g. \'{"context": {"$ilike": "%GENE%"}}\'',
    ),
    threshold: float | None = typer.Option(
        None, "--threshold", help="Minimum similarity score (0-1)"
    ),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
    table_name: str = typer.Option(
        ...,
        "--table-name",
        "-t",
        help="Table name for the vector store (must match ingest)",
    ),
    model: str = typer.Option(
        "openai/gpt-4.1-nano",
        "--model",
        "-m",
        help="LLM model name (used for resolver config)",
    ),
    embedder_name: str = typer.Option(
        "openai/text-embedding-3-small",
        "--embedder-name",
        "-e",
        help="Embedding model name (must match ingest)",
    ),
    model_api_key: str | None = typer.Option(
        None,
        "--model-api-key",
        help="LLM API key (default: OPENROUTER_API_KEY env)",
    ),
    model_temperature: float = typer.Option(
        0.2,
        "--model-temperature",
        help="LLM temperature",
    ),
    model_base_url: str | None = typer.Option(
        None,
        "--model-base-url",
        help="LLM base URL (default: OPENROUTER_BASE_URL env)",
    ),
    embedder_api_key: str | None = typer.Option(
        None,
        "--embedder-api-key",
        help="Embedder API key (default: OPENROUTER_API_KEY env)",
    ),
    embedder_base_url: str | None = typer.Option(
        None,
        "--embedder-base-url",
        help="Embedder base URL (default: OPENROUTER_BASE_URL env)",
    ),
) -> None:
    """Query the vector store for similar entities."""
    filter_dict: dict[str, Any] | None = None
    if filter_json is not None:
        try:
            filter_dict = json.loads(filter_json)
        except json.JSONDecodeError as e:
            typer.echo(f"Invalid --filter JSON: {e}", err=True)
            raise typer.Exit(code=1) from e

    param = SimilaritySearchParam(k=k, filter=filter_dict, threshold=threshold)

    api_key = model_api_key or os.getenv("OPENROUTER_API_KEY", "")
    base_url = model_base_url or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    embedder_api_key_val = embedder_api_key or api_key
    embedder_base_url_val = embedder_base_url or base_url
    resolver_config = ElementResolverConfig(
        llm=LLMConfig(
            name=model,
            temperature=model_temperature,
            api_key=SecretStr(api_key),
            base_url=base_url,
        ),
        db=PostgresDBConfig(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", "postgres"),
            database=os.getenv("POSTGRES_DATABASE", "hakken_agents"),
        ),
        table=VectorDBTableConfig(
            name=table_name,
            schema_name="public",
            content_column="content",
            embedding_column="embedding",
            metadata_columns=[
                Column(name="name", data_type="TEXT"),
                Column(name="description", data_type="TEXT"),
                Column(name="context", data_type="TEXT"),
                Column(name="element_id", data_type="TEXT"),
                Column(name="context_id", data_type="TEXT"),
            ],
        ),
        embedder=EmbedderConfig(
            embedding_model=embedder_name,
            embedding_dim=1536,
            api_key=SecretStr(embedder_api_key_val),
            base_url=embedder_base_url_val,
        ),
        content_fields=["name"],
        context_fields=["context"],
    )
    resolver = ElementResolver.from_config(resolver_config)

    results = resolver.find_similar_elements_with_score(query_text, param)

    if json_output:
        out = [
            {
                "element_id": doc.metadata.get("element_id", ""),
                "score": round(score, 4),
                "content": doc.page_content,
                "metadata": doc.metadata,
            }
            for doc, score in results
        ]
        typer.echo(json.dumps(out, indent=2))
    else:
        for doc, score in results:
            context_val = doc.metadata.get("context", "")
            typer.echo(f"{score:.2f} - {context_val}")
            typer.echo(doc.page_content)
            typer.echo("-" * 80)


if __name__ == "__main__":
    app()
