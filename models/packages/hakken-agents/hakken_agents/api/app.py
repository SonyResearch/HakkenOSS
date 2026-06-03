"""FastAPI application for the Element Resolver API."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from hakken_agents.api.config import ElementResolverAPIConfig
from hakken_agents.api.routers.element_resolver import router as element_resolver_router
from hakken_agents.tools.element_resolver import ElementResolver, TableRegistry


def create_app(config: ElementResolverAPIConfig) -> FastAPI:
    """Create the FastAPI app with the given config (e.g. from Hydra)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        table_name = config.resolver.table.name
        registry = TableRegistry(config.resolver.db)
        entry = await registry.get(table_name)

        if entry is not None:
            embedder_updates: dict = {
                "embedding_model": entry.embedder_model,
                "embedding_dim": entry.embedder_dim,
            }
            if entry.embedder_base_url is not None:
                embedder_updates["base_url"] = entry.embedder_base_url
            patched_embedder = config.resolver.embedder.model_copy(update=embedder_updates)
            patched_table = entry.to_table_config()
            resolver_config = config.resolver.model_copy(
                update={"embedder": patched_embedder, "table": patched_table}
            )
            logger.info(
                f"Registry: loaded config for table '{table_name}' "
                f"(embedder={entry.embedder_model}, dim={entry.embedder_dim}, "
                f"base_url={entry.embedder_base_url}, "
                f"columns={[c['name'] for c in entry.metadata_columns]})"
            )
        else:
            logger.warning(
                f"No registry entry for table '{table_name}'. "
                f"Using config as-is. Run ingest to register the table."
            )
            resolver_config = config.resolver

        app.state.element_resolver = ElementResolver.from_config(resolver_config)
        logger.info("Element resolver initialized")
        yield

    app = FastAPI(
        title="Hakken Agents API",
        description="Element ingestion and similarity search API",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(element_resolver_router, prefix="/api/v1")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    return app
