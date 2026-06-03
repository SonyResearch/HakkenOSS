import json
import uuid
from typing import Any, Self

from langchain.chat_models.base import BaseChatModel
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from tenacity import RetryCallState, retry, stop_after_attempt, wait_exponential

from hakken_agents.tools.element_resolver.config import ElementResolverConfig
from hakken_agents.tools.element_resolver.prompts import (
    DESCRIPTION_SYSTEM,
    render_description_user,
)
from hakken_agents.tools.element_resolver.schemas import SimilaritySearchParam
from hakken_agents.utils.llm import get_llm
from hakken_agents.vector_db.engine import VectorDBEngine


def _log_retry(retry_state: RetryCallState) -> None:
    wait = retry_state.next_action.sleep if retry_state.next_action else 0
    logger.warning(
        f"Retry #{retry_state.attempt_number} for {retry_state.fn.__name__} "
        f"after {retry_state.outcome.exception().__class__.__name__}: "
        f"{retry_state.outcome.exception()} — sleeping {wait:.0f}s"
    )


class ElementResolver:
    def __init__(
        self,
        llm: BaseChatModel,
        vector_db: VectorDBEngine,
    ) -> None:
        self.llm_description = llm

        self.vector_db = vector_db

        self.reserved_fields = ["id"]

    @retry(
        wait=wait_exponential(multiplier=10, min=10, max=120),
        stop=stop_after_attempt(5),
        before_sleep=_log_retry,
        reraise=True,
    )
    def get_description(self, document: Document) -> str:
        """Generate a short, searchable description of the content using the LLM."""

        user_content = render_description_user(document.page_content)
        messages = [
            SystemMessage(content=DESCRIPTION_SYSTEM),
            HumanMessage(content=user_content),
        ]
        response = self.llm_description.invoke(messages)
        return (response.content or "").strip()

    def validate_kwargs(self, **kwargs: Any) -> None:
        for field in self.reserved_fields:
            if field in kwargs:
                raise ValueError(f"{field} field is not allowed to be in the kwargs")

    def to_document(self, content: str, element_uuid: str | None = None, **kwargs: Any) -> Document:
        self.validate_kwargs(**kwargs)
        if element_uuid is None:
            element_uuid = self.get_uuid(**kwargs)
        return Document(id=element_uuid, page_content=content, metadata=kwargs)

    def add_description(self, document: Document) -> Document:
        if "description" in document.metadata:
            logger.warning("Description is already in document. Skipping")
            return document
        description = self.get_description(document)
        document.metadata["description"] = description
        logger.info(f"Description: {description}")
        document.page_content = f"{document.page_content}\n{description}"
        return document

    def to_document_with_description(
        self, content: str, element_uuid: str | None = None, **kwargs: Any
    ) -> Document:
        document = self.to_document(content, element_uuid, **kwargs)
        return self.add_description(document)

    # ------------------------------------------------------------------
    # Batch variants - parallelise LLM calls via BaseChatModel.batch()
    # ------------------------------------------------------------------

    @retry(
        wait=wait_exponential(multiplier=10, min=10, max=120),
        stop=stop_after_attempt(5),
        before_sleep=_log_retry,
        reraise=True,
    )
    def get_descriptions_batch(
        self, documents: list[Document], *, max_concurrency: int = 5
    ) -> list[str]:
        """Generate descriptions for multiple documents with parallel LLM calls."""
        messages_list = [
            [
                SystemMessage(content=DESCRIPTION_SYSTEM),
                HumanMessage(content=render_description_user(document.page_content)),
            ]
            for document in documents
        ]
        responses = self.llm_description.batch(
            messages_list, config={"max_concurrency": max_concurrency}
        )
        return [(r.content or "").strip() for r in responses]

    def add_descriptions_batch(
        self, documents: list[Document], *, max_concurrency: int = 5
    ) -> list[Document]:
        """Add LLM-generated descriptions to documents that lack one."""
        need_desc = [
            (i, doc) for i, doc in enumerate(documents) if "description" not in doc.metadata
        ]
        if not need_desc:
            return documents

        indices, docs_subset = zip(*need_desc, strict=True)
        descriptions = self.get_descriptions_batch(
            list(docs_subset), max_concurrency=max_concurrency
        )

        for idx, desc in zip(indices, descriptions, strict=True):
            documents[idx].metadata["description"] = desc
            documents[idx].page_content = f"{documents[idx].page_content}\n{desc}"
            logger.info(f"Description for {documents[idx].id}: {desc}")

        return documents

    def to_documents_with_description_batch(
        self,
        elements: list[dict[str, Any]],
        *,
        max_concurrency: int = 5,
    ) -> list[Document]:
        """Create Documents with LLM-generated descriptions from raw element dicts."""
        documents = [self.to_document(**element) for element in elements]
        return self.add_descriptions_batch(documents, max_concurrency=max_concurrency)

    def add(self, document: Document) -> str:
        if self.exists(document.id):
            logger.warning(f"Element {document.id} already exists. Skipping addition.")
            return document.id

        logger.info(f"Adding document: {document.id}")

        doc_id = self.vector_db.add_documents([document])[0]
        if doc_id != document.id:
            logger.warning(f"Document ID mismatch: {document.id} -> {doc_id}")
        return doc_id

    def get_filter_columns(self) -> list[str]:
        """Return the column names that can be used for filtering in similarity search.

        Queries the table schema to discover metadata columns (excludes standard
        langchain columns: langchain_id, content, embedding, langchain_metadata).
        """
        return self.vector_db.get_filter_columns()

    def exists(self, element_uuid: str) -> bool:
        """Return True if an element with the given UUID exists in the vector store."""
        return self.vector_db.exists(element_uuid)

    def add_many(
        self,
        documents: list[Document],
    ) -> list[str]:
        new_docs = []
        for document in documents:
            if self.exists(document.id):
                logger.warning(f"Element {document.id} already exists. Skipping.")
            else:
                new_docs.append(document)

        if new_docs:
            logger.info(f"Bulk-adding {len(new_docs)} document(s).")
            self.vector_db.add_documents(new_docs)

        return [doc.id for doc in documents]

    def get(self, uuid: str) -> Document:
        """Return the element document for the given UUID. Raises ValueError if not found."""
        docs = self.vector_db.get_by_ids([uuid])
        if not docs:
            raise ValueError(f"Document not found: {uuid}")
        return docs[0]

    def get_all(self) -> list[Document]:
        """Return all element documents from the vector store."""
        return self.vector_db.get_all_documents()

    def get_uuid(self, **kwargs: Any) -> str:
        payload = json.dumps(sorted(kwargs.items()), default=str)
        return str(uuid.uuid5(uuid.NAMESPACE_OID, payload))

    def find_similar_elements(self, query: str, param: SimilaritySearchParam) -> list[Document]:
        output = self.find_similar_elements_with_score(query, param)
        return [document for document, _ in output]

    def find_similar_elements_with_score(
        self, query: str, param: SimilaritySearchParam
    ) -> list[tuple[Document, float]]:
        """Find elements similar to the given query, ranked by similarity score.

        Searches the vector store using the query text. Returns similarity scores
        computed as 1.0 - distance (higher = more similar). When ``param.threshold``
        is set, only results with a score >= threshold are returned.

        Args:
            query: The search query text.
            param: Search parameters (k, threshold, filter).

        Returns:
            List of (Document, similarity_score) tuples sorted by similarity
            (higher score = more similar).
        """
        results = self.vector_db.similarity_search_with_score(
            query=query,
            k=param.k,
            filter=param.filter,
        )
        scored = [(doc, 1.0 - distance) for doc, distance in results]
        if param.threshold is not None:
            scored = [(doc, score) for doc, score in scored if score >= param.threshold]
        return scored

    @classmethod
    def from_config(cls, config: ElementResolverConfig) -> Self:
        llm = get_llm(config.llm)
        vector_db = VectorDBEngine.create_from_config(config.vector_db)
        return cls(
            llm=llm,
            vector_db=vector_db,
        )
