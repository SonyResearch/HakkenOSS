from collections.abc import Sequence
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGEngine
from langchain_postgres.v2.engine import Column
from langchain_postgres.v2.vectorstores import PGVectorStore
from loguru import logger
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from hakken_agents.db.actions.postgres import PostgresActions
from hakken_agents.vector_db.config import VectorDBConfig
from hakken_agents.vector_db.enums import SimilarityMetric
from hakken_agents.vector_db.similarity_metrics import compute_similarity_matrix


class VectorDBEngine:
    def __init__(
        self,
        engine: PGEngine,
        embedding_service: Embeddings,
    ) -> None:
        self._engine = engine
        self._embedding_service = embedding_service
        self._postgres_actions = PostgresActions.from_engine(engine._pool)
        self._store: PGVectorStore | None = None
        self._table_name: str | None = None
        self._schema_name: str | None = None
        vec = self._embedding_service.embed_query("hello")
        self.embedding_dim = len(vec)

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string (sync)."""
        return self._embedding_service.embed_query(query)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of text strings (sync)."""
        return self._embedding_service.embed_documents(texts)

    async def aembed_query(self, query: str) -> list[float]:
        """Embed a single query string (async)."""
        return await self._embedding_service.aembed_query(query)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of text strings (async)."""
        return await self._embedding_service.aembed_documents(texts)

    def table_exists(self, table_name: str, schema_name: str = "public") -> bool:
        """Check if a table exists in the database (sync version)."""
        return self._engine._run_as_sync(
            self._postgres_actions.table_exists(table_name, schema=schema_name)
        )

    async def atable_exists(self, table_name: str, schema_name: str = "public") -> bool:
        """Check if a table exists in the database (async version)."""
        return await self._postgres_actions.table_exists(table_name, schema=schema_name)

    async def ainit_vectorstore_table(
        self,
        table_name: str,
        schema_name: str = "public",
        content_column: str = "content",
        embedding_column: str = "embedding",
        metadata_columns: list[Column] | None = None,
    ) -> None:
        table_exists = await self.atable_exists(table_name, schema_name)
        if table_exists:
            logger.warning(
                f"Table {table_name} already exists in schema {schema_name}. "
                "Skipping initialization."
            )
            return

        await self._engine.ainit_vectorstore_table(
            table_name=table_name,
            vector_size=self.embedding_dim,
            schema_name=schema_name,
            content_column=content_column,
            embedding_column=embedding_column,
            metadata_columns=metadata_columns,
        )

    async def acreate(
        self,
        table_name: str,
        schema_name: str = "public",
        metadata_columns: list[str] | None = None,
    ) -> None:
        self._table_name = table_name
        self._schema_name = schema_name
        self._store = await PGVectorStore.create(
            engine=self._engine,
            embedding_service=self._embedding_service,
            table_name=table_name,
            schema_name=schema_name,
            metadata_columns=metadata_columns,
        )

    async def aexists(self, doc_id: str, id_column: str = "langchain_id") -> bool:
        """Return True if a document with the given id exists in the vector store.

        Requires acreate() to have been called so that table and schema are set.

        Args:
            doc_id: The document id (e.g. element UUID).
            id_column: Name of the id column. Defaults to "langchain_id".

        Returns:
            True if the document exists, False otherwise (or if store/table not set).
        """
        if self._table_name is None or self._schema_name is None:
            return False
        if not id_column.replace("_", "").isalnum():
            raise ValueError("id_column must be alphanumeric with underscores only")
        query = (
            f'SELECT 1 FROM "{self._schema_name}"."{self._table_name}" '
            f'WHERE "{id_column}" = :doc_id LIMIT 1'
        )
        result = await self._postgres_actions.execute_query(query, {"doc_id": doc_id})
        return len(result) > 0

    def exists(self, doc_id: str, id_column: str = "langchain_id") -> bool:
        """Sync version of aexists."""
        return self._engine._run_as_sync(self.aexists(doc_id, id_column=id_column))

    async def aget_by_ids(self, ids: Sequence[str]) -> list[Document]:
        """Get documents by their IDs (async).

        Delegates to the underlying store. Follows the LangChain VectorStore
        contract: returns only documents that exist; does not raise if some IDs
        are missing. Returns [] if store not initialized or ids is empty.
        """
        if self._store is None or not ids:
            return []
        return await self._store.aget_by_ids(list(ids))

    def get_by_ids(self, ids: Sequence[str]) -> list[Document]:
        """Sync version of aget_by_ids. Get documents by their IDs."""
        if self._store is None or not ids:
            return []
        return self._store.get_by_ids(list(ids))

    async def aadd_documents(self, documents: list[Document]) -> list[str]:
        return await self._store.aadd_documents(documents)

    def add_documents(self, documents: list[Document]) -> list[str]:
        return self._store.add_documents(documents)

    async def asimilarity_search(
        self, query: str, k: int = 4, filter: dict | None = None, **kwargs: Any
    ) -> list[Document]:
        """Return the k most similar documents to the query (async).

        Args:
            query: Query string to search for.
            k: Number of documents to return. Defaults to 4.

        Returns:
            List of Document objects most similar to the query.
        """
        return await self._store.asimilarity_search(query, k=k, filter=filter, **kwargs)

    async def asimilarity_search_with_score(
        self,
        query: str,
        k: int | None = None,
        filter: dict | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Return the k most similar documents to the query with distance scores (async).

        Args:
            query: Query string to search for.
            k: Number of documents to return. Defaults to store default (e.g. 4).
            filter: Optional metadata filter for the search.

        Returns:
            List of (Document, score) tuples. Score is a distance (lower is more similar).
        """
        return await self._store.asimilarity_search_with_score(query, k=k, filter=filter, **kwargs)

    def similarity_search_with_score(
        self,
        query: str,
        k: int | None = None,
        filter: dict | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Sync version of asimilarity_search_with_score."""
        return self._store.similarity_search_with_score(query, k=k, filter=filter, **kwargs)

    async def acount_documents(
        self,
        table_name: str,
        schema_name: str = "public",
    ) -> int:
        """Count the number of documents in a table.

        Args:
            table_name: Name of the table to count documents from.
            schema_name: Schema containing the table. Defaults to "public".

        Returns:
            The total number of documents stored.
        """
        query = f"""
            SELECT COUNT(*) FROM "{schema_name}"."{table_name}"
        """
        result = await self._postgres_actions.execute_query(query)
        return result[0][0] if result else 0

    async def asimilarity_scores(
        self,
        query_list_1: list[str],
        query_list_2: list[str] | None = None,
        metric: SimilarityMetric = SimilarityMetric.COSINE,
    ) -> list[list[float]]:
        """Compute pairwise similarity scores between all queries.

        Args:
            query_list: List of query strings to compute similarities for.
            metric: Similarity metric (COSINE, EUCLIDEAN, or DOT_PRODUCT).
                   Defaults to COSINE.

        Returns:
            A matrix of size (len(query_list), len(query_list)) where element [i][j]
            contains the similarity between query_list[i] and query_list[j].
        """
        embeddings_1 = await self.aembed_documents(query_list_1)
        embeddings_2 = (
            await self.aembed_documents(query_list_2) if query_list_2 is not None else None
        )

        return compute_similarity_matrix(embeddings_1, embeddings_2=embeddings_2, metric=metric)

    async def alist_documents(
        self,
        table_name: str,
        schema_name: str = "public",
        limit: int | None = None,
        offset: int = 0,
        content_column: str = "content",
        id_column: str = "langchain_id",
    ) -> list[Document]:
        """List documents from a table.

        Args:
            table_name: Name of the table to list documents from.
            schema_name: Schema containing the table. Defaults to "public".
            limit: Maximum number of documents to return. If None, returns all.
            offset: Number of documents to skip (for pagination).
            content_column: Name of the content column. Defaults to "content".
            id_column: Name of the ID column. Defaults to "langchain_id".

        Returns:
            List of Document objects with id, page_content, and flattened metadata.
        """
        # Build query to fetch all columns except embedding (too large)
        columns_query = f"""
            SELECT column_name FROM information_schema.columns
            WHERE table_schema = '{schema_name}'
            AND table_name = '{table_name}'
            AND data_type != 'USER-DEFINED'
            ORDER BY ordinal_position
        """
        columns_result = await self._postgres_actions.execute_query(columns_query)
        columns = [row[0] for row in columns_result]

        # Build the main query
        columns_str = ", ".join(f'"{col}"' for col in columns)
        query = f"""
            SELECT {columns_str}
            FROM "{schema_name}"."{table_name}"
            ORDER BY "{id_column}"
        """
        if limit is not None:
            query += f" LIMIT {limit}"
        if offset > 0:
            query += f" OFFSET {offset}"

        result = await self._postgres_actions.execute_query(query)

        # Convert to Documents
        documents = []
        for row in result:
            row_dict = dict(zip(columns, row, strict=False))
            content = row_dict.pop(content_column, "")
            # Extract ID for Document.id field
            doc_id = row_dict.pop(id_column, None)
            if doc_id is not None:
                doc_id = str(doc_id)
            # Flatten langchain_metadata into metadata if present
            langchain_metadata = row_dict.pop("langchain_metadata", None)
            if isinstance(langchain_metadata, dict):
                row_dict.update(langchain_metadata)
            documents.append(Document(id=doc_id, page_content=str(content), metadata=row_dict))

        return documents

    async def aget_all_documents(self) -> list[Document]:
        """Return all documents from the current table. Returns [] if store not initialized."""
        if self._table_name is None or self._schema_name is None:
            return []
        return await self.alist_documents(
            table_name=self._table_name,
            schema_name=self._schema_name,
        )

    def get_all_documents(self) -> list[Document]:
        """Sync version of aget_all_documents."""
        return self._engine._run_as_sync(self.aget_all_documents())

    _STANDARD_COLUMNS = frozenset({"langchain_id", "content", "embedding", "langchain_metadata"})

    async def aget_filter_columns(
        self,
        table_name: str | None = None,
        schema_name: str = "public",
    ) -> list[str]:
        """Return column names that can be used for filtering (metadata columns).

        Queries the table schema and excludes standard langchain columns
        (langchain_id, content, embedding, langchain_metadata).

        Args:
            table_name: Table name. Uses the store's table if None and store is initialized.
            schema_name: Schema name. Uses the store's schema if table_name is None.

        Returns:
            List of filterable column names. Empty if table not found or not initialized.
        """
        tbl = table_name or self._table_name
        schema = self._schema_name if table_name is None and self._table_name else schema_name
        if tbl is None:
            return []
        columns_query = """
            SELECT column_name FROM information_schema.columns
            WHERE table_schema = :schema_name AND table_name = :table_name
            AND data_type != 'USER-DEFINED'
            ORDER BY ordinal_position
        """
        result = await self._postgres_actions.execute_query(
            columns_query,
            {"schema_name": schema, "table_name": tbl},
        )
        columns = [row[0] for row in result]
        return [c for c in columns if c not in self._STANDARD_COLUMNS]

    def get_filter_columns(
        self,
        table_name: str | None = None,
        schema_name: str = "public",
    ) -> list[str]:
        """Sync version of aget_filter_columns."""
        return self._engine._run_as_sync(
            self.aget_filter_columns(table_name=table_name, schema_name=schema_name)
        )

    async def adelete_all_documents(
        self,
        table_name: str,
        schema_name: str = "public",
    ) -> int:
        """Delete all documents from a table.

        Args:
            table_name: Name of the table to delete documents from.
            schema_name: Schema containing the table. Defaults to "public".

        Returns:
            The number of documents deleted.
        """
        count = await self.acount_documents(table_name, schema_name)

        engine = await self._postgres_actions._get_engine()
        async with engine.connect() as conn:
            query = text(f'DELETE FROM "{schema_name}"."{table_name}"')
            await conn.execute(query)
            await conn.commit()

        return count

    @classmethod
    async def acreate_from_config(cls, config: VectorDBConfig) -> "VectorDBEngine":
        # Use from_engine() with loop=None to avoid event loop conflicts in async contexts
        # (e.g., Jupyter notebooks). This ensœures coroutines run in the current loop
        # rather than being dispatched to a background thread's loop.
        async_engine = create_async_engine(config.db.get_connection_string(True))
        engine = PGEngine.from_engine(async_engine, loop=None)

        embedding_service = OpenAIEmbeddings(
            model=config.embedder.embedding_model,
            api_key=config.embedder.api_key.get_secret_value(),
            base_url=config.embedder.base_url,
        )
        vector_db = cls(engine, embedding_service)

        if config.table is not None:
            metadata_col_names = (
                [col.name for col in config.table.metadata_columns]
                if config.table.metadata_columns
                else None
            )
            await vector_db.ainit_vectorstore_table(
                table_name=config.table.name,
                schema_name=config.table.schema_name,
                content_column=config.table.content_column,
                embedding_column=config.table.embedding_column,
                metadata_columns=config.table.metadata_columns,
            )
            await vector_db.acreate(
                config.table.name,
                config.table.schema_name,
                metadata_columns=metadata_col_names,
            )

        return vector_db

    @classmethod
    def create_from_config(cls, config: VectorDBConfig) -> "VectorDBEngine":
        # Use from_connection_string so PGEngine gets a background event loop;
        # from_engine(loop=None) would leave _run_as_sync() with no loop and raise.
        engine = PGEngine.from_connection_string(config.db.get_connection_string(True))
        if len(config.embedder.api_key.get_secret_value()) > 0:
            embedding_service = OpenAIEmbeddings(
                model=config.embedder.embedding_model,
                api_key=config.embedder.api_key.get_secret_value(),
                base_url=config.embedder.base_url,
            )
        else:
            embedding_service = OllamaEmbeddings(
                model=config.embedder.embedding_model,
                base_url=config.embedder.base_url,
            )

        vector_db = cls(engine, embedding_service)

        if config.table is not None:
            metadata_col_names = (
                [col.name for col in config.table.metadata_columns]
                if config.table.metadata_columns
                else None
            )
            engine._run_as_sync(
                vector_db.ainit_vectorstore_table(
                    table_name=config.table.name,
                    schema_name=config.table.schema_name,
                    content_column=config.table.content_column,
                    embedding_column=config.table.embedding_column,
                    metadata_columns=config.table.metadata_columns,
                )
            )
            engine._run_as_sync(
                vector_db.acreate(
                    config.table.name,
                    config.table.schema_name,
                    metadata_columns=metadata_col_names,
                )
            )

        return vector_db
