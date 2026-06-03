import json
import uuid as uuid_module

from langchain_core.documents import Document
from loguru import logger

from hakken_agents.enki.db.facts_table import FactsTable
from hakken_agents.enki.schemas.fact import Fact
from hakken_agents.enki.schemas.relation import Relation
from hakken_agents.tools.element_resolver import ElementResolver
from hakken_agents.tools.element_resolver.schemas import SimilaritySearchParam
from hakken_agents.utils.llm import get_llm
from hakken_agents.vector_db.engine import VectorDBEngine

from .config import FactResolverConfig


class FactResolver:
    """Resolves extracted facts by linking entities, deduplicating relations,
    and storing the resolved triples in a relational table.

    Every resolved artifact (entity, relation, fact) is a plain ``Document``.

    - **Entity matching**: exact ``(name, domain)`` lookup against the
      already-resolved entity Documents passed in by the caller.
    - **Relation resolution**: deduplication via an internal ``ElementResolver``
      backed by the ``relations_vectors`` table.
    - **Fact storage**: delegated to ``FactsTable`` (sync, pooled, with
      UNIQUE constraint for natural dedup).
    """

    def __init__(
        self,
        relation_resolver: ElementResolver,
        facts_table: FactsTable,
    ) -> None:
        self._relation_resolver = relation_resolver
        self._facts_table = facts_table

    @property
    def relation_resolver(self) -> ElementResolver:
        return self._relation_resolver

    @staticmethod
    def build_entity_lookup(
        entities: list[Document],
    ) -> dict[tuple[str, str], Document]:
        """Build a ``(name, domain) -> Document`` lookup from resolved entities."""
        return {(doc.metadata["name"], doc.metadata["domain"]): doc for doc in entities}

    @staticmethod
    def _get_fact_uuid(
        subject_id: str,
        relation_id: str,
        object_id: str,
        chunk_uuid: str | None = None,
    ) -> str:
        """Deterministic UUID for a fact occurrence (triple + source chunk)."""
        payload = json.dumps(
            [
                ("subject_id", subject_id),
                ("relation_id", relation_id),
                ("object_id", object_id),
                ("chunk_uuid", chunk_uuid),
            ],
            default=str,
        )
        return str(uuid_module.uuid5(uuid_module.NAMESPACE_OID, payload))

    def resolve_relation(self, relation: Relation) -> Document:
        """Resolve a relation, deduplicating by name (one row per relation name)."""
        doc = self._relation_resolver.to_document(name=relation.name)
        if self._relation_resolver.exists(doc.id):
            logger.debug(f"Relation '{relation.name}' exists: {doc.id}")
            return self._relation_resolver.get(doc.id)

        self._relation_resolver.add(doc)
        logger.info(f"New relation '{relation.name}' added: {doc.id}")
        return doc

    def resolve_fact(
        self,
        fact: Fact,
        entity_lookup: dict[tuple[str, str], Document],
        chunk_uuid: str | None = None,
    ) -> Document | None:
        """Resolve a single fact into a fully linked ``Document``.

        Performs three lookups:
        1. Match ``(subject_name, subject_domain)`` to a resolved entity.
        2. Match ``(object_name, object_domain)`` to a resolved entity.
        3. Deduplicate the relation via the internal ``ElementResolver``.

        Args:
            fact: Raw fact from the FactExtractor.
            entity_lookup: Pre-built ``(name, domain) -> Document`` mapping.
            chunk_uuid: Optional source chunk UUID for provenance.

        Returns:
            A ``Document`` with resolved UUIDs in metadata, or ``None`` if
            subject or object cannot be matched (logged as a warning).
        """
        subject_key = (fact.subject_name, fact.subject_domain)
        subject_doc = entity_lookup.get(subject_key)
        if subject_doc is None:
            logger.warning(f"Subject not found in resolved entities: {subject_key}")
            return None

        object_key = (fact.object_name, fact.object_domain)
        object_doc = entity_lookup.get(object_key)
        if object_doc is None:
            logger.warning(f"Object not found in resolved entities: {object_key}")
            return None

        relation_doc = self.resolve_relation(fact.relation)

        fact_id = self._get_fact_uuid(subject_doc.id, relation_doc.id, object_doc.id, chunk_uuid)
        page_content = (
            f"{subject_doc.metadata['name']} "
            f"--[{relation_doc.metadata['name']}]--> "
            f"{object_doc.metadata['name']}"
        )
        metadata: dict = {
            "subject_id": subject_doc.id,
            "subject_name": subject_doc.metadata["name"],
            "subject_domain": subject_doc.metadata["domain"],
            "relation_id": relation_doc.id,
            "relation_name": relation_doc.metadata["name"],
            "object_id": object_doc.id,
            "object_name": object_doc.metadata["name"],
            "object_domain": object_doc.metadata["domain"],
            "chunk_uuid": chunk_uuid,
        }
        if fact.subject_quantity:
            metadata["subject_quantity"] = fact.subject_quantity.model_dump()
        if fact.object_quantity:
            metadata["object_quantity"] = fact.object_quantity.model_dump()

        return Document(id=fact_id, page_content=page_content, metadata=metadata)

    def resolve_and_store_facts(
        self,
        facts: list[Fact],
        resolved_entities: list[Document],
        chunk_uuid: str | None = None,
    ) -> list[Document]:
        """Resolve and store a batch of facts.

        Args:
            facts: Raw facts from the FactExtractor.
            resolved_entities: Already-resolved entity Documents (with UUIDs).
            chunk_uuid: Optional source chunk UUID for provenance.

        Returns:
            List of successfully resolved and stored fact Documents.
        """
        entity_lookup = self.build_entity_lookup(resolved_entities)

        resolved: list[Document] = []
        for fact in facts:
            fact_doc = self.resolve_fact(fact, entity_lookup, chunk_uuid)
            if fact_doc is not None:
                inserted_id = self._facts_table.insert_fact(fact_doc)
                if inserted_id is None:
                    logger.debug(f"Duplicate fact skipped: {fact_doc.page_content}")
                resolved.append(fact_doc)

        logger.info(f"Resolved {len(resolved)}/{len(facts)} facts")
        return resolved

    def seed_preferred_relations(self, relation_names: list[str]) -> int:
        """Pre-seed the relation vector store with preferred relation type names.

        Each name is stored once; similarity search can retrieve them from the
        first chunk onward.

        Returns the number of newly added relations (skips duplicates).
        """
        added = 0
        for name in relation_names:
            doc = self._relation_resolver.to_document(name=name)
            if not self._relation_resolver.exists(doc.id):
                self._relation_resolver.add(doc)
                added += 1
        logger.info(f"Seeded {added}/{len(relation_names)} preferred relation types")
        return added

    def find_relevant_relation_types(
        self,
        query: str,
        param: SimilaritySearchParam,
    ) -> list[Document]:
        """Similarity-search the relation vector store for types relevant to *query*."""
        return self._relation_resolver.find_similar_elements(query, param)

    @classmethod
    def from_config(cls, config: FactResolverConfig) -> "FactResolver":
        llm = get_llm(config.relation_resolver.llm)
        vector_db = VectorDBEngine.create_from_config(config.relation_resolver.vector_db)
        relation_resolver = ElementResolver(
            llm=llm,
            vector_db=vector_db,
            content_fields=config.relation_resolver.content_fields,
            context_fields=config.relation_resolver.context_fields,
        )
        facts_table = FactsTable(db_config=config.db, table_name=config.facts_table)
        return cls(
            relation_resolver=relation_resolver,
            facts_table=facts_table,
        )
