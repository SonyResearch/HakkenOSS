from collections import defaultdict
from typing import TYPE_CHECKING

from complex_query.core.contracts.kg_ledger import KnowledgeGraphLedger
from complex_query.core.entities.config.kg_ledger import InMemoryKGLedgerConfig

if TYPE_CHECKING:
    from collections.abc import Sequence

    from query_common.entities.kg.concept import Concept
    from query_common.entities.kg.identifier import (
        ConceptIdentifier,
        DomainIdentifier,
        RelationIdentifier,
    )
    from query_common.entities.kg.triple import Triple


class InMemoryKnowledgeGraphLedger(KnowledgeGraphLedger[InMemoryKGLedgerConfig]):
    def __init__(self, config: InMemoryKGLedgerConfig) -> None:
        super().__init__(config)

        self._concept_cache: dict[ConceptIdentifier, Concept] = {}
        self._domain_concepts_cache: dict[DomainIdentifier, set[ConceptIdentifier]] = defaultdict(
            lambda: set()
        )

    def add_concept(self, concept: "Concept") -> None:
        self._concept_cache[concept.identifier] = concept

    def add_concepts_for_domain(
        self, concepts: "Sequence[Concept]", domain_identifier: str
    ) -> None:
        for concept in concepts:
            if domain_identifier != concept.domain_identifier:
                raise ValueError(
                    "Domain identifiers do not match, "
                    f"concept identifier: {concept.identifier}, "
                    f"concept domain identifier: {concept.domain_identifier}, "
                    f"domain_identifier given: {domain_identifier}"
                )

            self._concept_cache[concept.identifier] = concept
            # `self._domain_concepts_cache` is only updated in this method,
            # to avoid returning incomplete concepts when `get_concepts_from_domain` is called.
            self._domain_concepts_cache[domain_identifier].add(concept.identifier)

    def _get_concept(self, concept_identifier: "ConceptIdentifier") -> "Concept":
        if concept_identifier in self._concept_cache:
            return self._concept_cache[concept_identifier]
        raise KeyError(f"Concept with id {concept_identifier} not found")

    def get_concepts_from_domain(self, domain_identifier: "DomainIdentifier") -> list["Concept"]:
        if domain_identifier not in self._domain_concepts_cache:
            raise KeyError(f"Domain {domain_identifier} is not available in ledger")

        concept_identifiers_of_domain = self._domain_concepts_cache[domain_identifier]
        return [self._concept_cache[identifier] for identifier in concept_identifiers_of_domain]

    def add_triple(self, triple: "Triple") -> None:
        raise NotImplementedError

    def _get_triples(
        self,
        subject_identifier: "ConceptIdentifier | None" = None,
        object_identifier: "ConceptIdentifier | None" = None,
        relation_identifier: "RelationIdentifier | None" = None,
    ) -> list["Triple"]:
        raise NotImplementedError
