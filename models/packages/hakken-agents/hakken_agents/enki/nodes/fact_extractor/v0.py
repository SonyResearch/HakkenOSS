from typing import Any

from langchain_core.documents import Document
from loguru import logger

from hakken_agents.tools.info_extractor import InfoExtractor

from .config import FactExtractorConfig
from .schemas import ExtractedFacts


class FactExtractor(InfoExtractor[ExtractedFacts, FactExtractorConfig]):
    """Extracts structured facts (subject-relation-object triples) from text.

    Requires a list of resolved entity Documents so the LLM constrains its
    output to known entities only.  Both ``run`` and ``arun`` accept a typed
    ``entities`` argument instead of a raw ``user_variables`` dict.

    Optionally accepts ``preferred_relation_types`` (static allow-list, soft
    guidance) and ``relevant_relation_types`` (dynamic similarity-search
    results) to steer the LLM towards a canonical relation vocabulary.
    """

    output_schema = ExtractedFacts

    @property
    def allowed_user_variables(self) -> list[str]:
        return [
            "entities",
            "previous_text",
            "preferred_relation_types",
            "relevant_relation_types",
        ]

    @property
    def user_variables_are_required(self) -> bool:
        return True

    @staticmethod
    def format_entities(entities: list[Document]) -> str:
        """Format resolved entity Documents into the prompt representation.

        Each entity becomes a line of ``name || domain``.
        """
        return "\n".join(f"{doc.metadata['name']} || {doc.metadata['domain']}" for doc in entities)

    def _build_user_variables(
        self,
        entities: list[Document],
        previous_text: str | None = None,
        preferred_relation_types: str | None = None,
        relevant_relation_types: str | None = None,
    ) -> dict[str, Any]:
        if not entities:
            raise ValueError("entities must be a non-empty list for fact extraction")
        variables: dict[str, Any] = {
            "entities": self.format_entities(entities),
        }
        if previous_text is not None:
            variables["previous_text"] = previous_text
        if preferred_relation_types is not None:
            variables["preferred_relation_types"] = preferred_relation_types
        if relevant_relation_types is not None:
            variables["relevant_relation_types"] = relevant_relation_types
        return variables

    @staticmethod
    def _log_non_preferred_relations(
        result: ExtractedFacts,
        preferred_set: set[str] | None,
    ) -> None:
        """Soft warning: log any relation names not in the preferred set."""
        if preferred_set is None:
            return
        for fact in result.facts:
            if fact.relation.name not in preferred_set:
                logger.warning(f"Relation '{fact.relation.name}' is not in the preferred set")

    def run(
        self,
        text: str,
        entities: list[Document],
        previous_text: str | None = None,
        preferred_relation_types: str | None = None,
        relevant_relation_types: str | None = None,
    ) -> ExtractedFacts:
        user_variables = self._build_user_variables(
            entities,
            previous_text,
            preferred_relation_types,
            relevant_relation_types,
        )
        result = super().run(text, user_variables)
        if preferred_relation_types is not None:
            preferred_set = {t.strip() for t in preferred_relation_types.split("|")}
            self._log_non_preferred_relations(result, preferred_set)
        return result

    async def arun(
        self,
        text: str,
        entities: list[Document],
        previous_text: str | None = None,
        preferred_relation_types: str | None = None,
        relevant_relation_types: str | None = None,
    ) -> ExtractedFacts:
        user_variables = self._build_user_variables(
            entities,
            previous_text,
            preferred_relation_types,
            relevant_relation_types,
        )
        result = await super().arun(text, user_variables)
        if preferred_relation_types is not None:
            preferred_set = {t.strip() for t in preferred_relation_types.split("|")}
            self._log_non_preferred_relations(result, preferred_set)
        return result
