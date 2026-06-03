import json
import uuid as uuid_module
from typing import Any

from langchain_core.documents import Document
from loguru import logger

from hakken_agents.tools.element_resolver import ElementResolver
from hakken_agents.tools.element_resolver.schemas import SimilaritySearchParam
from hakken_agents.utils.llm import get_llm
from hakken_agents.vector_db.engine import VectorDBEngine

from .config import EntityResolverConfig


class EntityResolver(ElementResolver):
    """Resolves entities by deduplicating within the same domain via vector similarity.

    Resolution logic (in order):
    1. **Exact match** — deterministic UUID derived from (name, domain). If the
       entity already exists under that UUID, return the stored document immediately.
    2. **Near match** — similarity search over name+description embeddings,
       filtered to the same ``domain_id``. If the top candidate exceeds
       ``threshold``, return it.
    3. **New entity** — no match found; add the document to the store.
    """

    def __init__(
        self,
        *args,
        threshold: float = 0.95,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.threshold = threshold

    def get_uuid(self, **kwargs: Any) -> str:
        """Deterministic UUID based on (name, domain) — independent of description."""
        identity = {"name": kwargs["name"], "domain": kwargs["domain"]}
        payload = json.dumps(sorted(identity.items()), default=str)
        return str(uuid_module.uuid5(uuid_module.NAMESPACE_OID, payload))

    def resolve_entity(self, entity: Document) -> Document:
        """Resolve an entity document against the vector store.

        The document must have been built via ``to_document(name=...,
        description=..., domain=..., domain_id=...)`` so that ``id``,
        ``page_content``, and metadata are all populated correctly.

        Args:
            entity: Entity as a Document (id, page_content, metadata with
                    name, description, domain, domain_id).

        Returns:
            The matched or newly created Document.
        """
        if self.exists(entity.id):
            logger.debug(f"Entity '{entity.metadata.get('name')}' exact match: {entity.id}")
            return self.get(entity.id)

        candidates = self.find_similar_elements_with_score(
            entity.page_content,
            param=SimilaritySearchParam(
                k=1,
                threshold=self.threshold,
                filter={"domain_id": entity.metadata["domain_id"]},
            ),
        )

        if candidates:
            matched_doc, score = candidates[0]
            logger.info(
                f"Entity '{entity.metadata.get('name')}' matched "
                f"'{matched_doc.metadata.get('name')}' (score={score:.3f})"
            )
            return matched_doc

        self.add(entity)
        logger.info(f"New entity '{entity.metadata.get('name')}' added: {entity.id}")
        return entity

    @classmethod
    def from_config(cls, config: EntityResolverConfig) -> "EntityResolver":
        llm = get_llm(config.llm)
        vector_db = VectorDBEngine.create_from_config(config.vector_db)
        return cls(
            llm=llm,
            vector_db=vector_db,
            content_fields=config.content_fields,
            context_fields=config.context_fields,
            threshold=config.threshold,
        )
