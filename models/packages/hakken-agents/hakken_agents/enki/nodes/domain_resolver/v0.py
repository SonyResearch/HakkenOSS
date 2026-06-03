from langchain_core.documents import Document
from loguru import logger

from hakken_agents.tools.element_resolver import ElementResolver
from hakken_agents.tools.element_resolver.schemas import SimilaritySearchParam
from hakken_agents.utils.llm import get_llm
from hakken_agents.vector_db.engine import VectorDBEngine

from .config import DomainResolverConfig


class DomainResolver(ElementResolver):
    def __init__(
        self,
        *args,
        threshold: float = 0.80,
        fallback_threshold: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.threshold = threshold
        self.fallback_threshold = fallback_threshold

    def split_by_levels(self, domain: str) -> dict[str, str]:
        """Split a domain name into its levels."""
        return {f"level_{i + 1}": level for i, level in enumerate(domain.split("/"))}

    def resolve_domain(
        self,
        domain: str,
        allowed_domains: list[str] | None = None,
        fallback_threshold: float | None = None,
    ) -> Document | None:
        """Resolve a domain name, deduplicating against existing domains.

        When allowed_domains is set and domain is not in the list, performs
        similarity search among allowed domains and returns the closest match
        if its score >= fallback_threshold; otherwise returns None (caller should
        skip the entity).

        Returns the matched or newly created Document, or None if the entity
        should be dropped (disallowed domain with no good allowed match).
        """
        allowed_set = set(allowed_domains) if allowed_domains else None
        thresh = fallback_threshold if fallback_threshold is not None else self.fallback_threshold

        if allowed_set is not None and domain not in allowed_set:
            return self._resolve_to_closest_allowed(domain, allowed_set, thresh)

        levels = self.split_by_levels(domain)
        doc = self.to_document(name=domain, **levels)
        if self.exists(doc.id):
            logger.debug(f"Domain exists with id {doc.id}")
            return self.get(doc.id)

        query = self.get_content(name=domain)
        similar_docs = self.find_similar_elements_with_score(
            query, SimilaritySearchParam(k=1, filter=None)
        )
        if len(similar_docs) > 0:
            similar_doc, score = similar_docs[0]
            if score > self.threshold and self.docs_are_similar(doc, similar_doc):
                return similar_doc

        self.add(doc)
        return doc

    def _resolve_to_closest_allowed(
        self, domain: str, allowed_set: set[str], fallback_threshold: float
    ) -> Document | None:
        """Find the closest allowed domain by similarity; return None if below threshold."""
        query = self.get_content(name=domain)
        k = max(50, len(allowed_set) * 2)
        similar_docs = self.find_similar_elements_with_score(
            query, SimilaritySearchParam(k=k, filter=None)
        )
        allowed_matches = [
            (doc, score) for doc, score in similar_docs if doc.metadata.get("name") in allowed_set
        ]
        if not allowed_matches:
            logger.info(
                f"Domain {domain!r} not in allowed list and no allowed match in top-{k}; "
                "dropping entity"
            )
            return None
        best_doc, best_score = allowed_matches[0]
        if best_score < fallback_threshold:
            logger.info(
                f"Domain {domain!r} closest allowed match {best_doc.metadata.get('name')!r} "
                f"(score={best_score:.3f}) below threshold {fallback_threshold}; dropping entity"
            )
            return None
        logger.info(
            f"Domain {domain!r} not in allowed list; reassigned to closest "
            f"{best_doc.metadata.get('name')!r} (score={best_score:.3f})"
        )
        return best_doc

    def docs_are_similar(self, doc_1: Document, doc_2: Document) -> bool:
        """Return True if both documents share the same parent levels."""
        keys = ("level_1", "level_2", "level_3", "level_4")
        levels_1 = [doc_1.metadata.get(k, "") for k in keys]
        levels_2 = [doc_2.metadata.get(k, "") for k in keys]
        parent_1 = [v for v in levels_1 if v][:-1]
        parent_2 = [v for v in levels_2 if v][:-1]
        return parent_1 == parent_2

    @classmethod
    def from_config(cls, config: DomainResolverConfig) -> "DomainResolver":
        llm = get_llm(config.llm)
        domains_vdb = VectorDBEngine.create_from_config(config.vector_db)
        return cls(
            llm=llm,
            vector_db=domains_vdb,
            content_fields=config.content_fields,
            context_fields=config.context_fields,
            threshold=config.threshold,
            fallback_threshold=config.fallback_threshold,
        )
