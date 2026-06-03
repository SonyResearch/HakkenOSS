from __future__ import annotations

from typing import TYPE_CHECKING

import h5py
from loguru import logger
from query_common.entities.kg.concept import Concept

from complex_query.core.contracts.kg_ledger import KnowledgeGraphLedger
from complex_query.core.entities.config.kg_ledger import HDF5KGLedgerConfig

if TYPE_CHECKING:
    from collections.abc import Sequence

    from query_common.entities.kg.identifier import (
        ConceptIdentifier,
        DomainIdentifier,
        RelationIdentifier,
    )
    from query_common.entities.kg.triple import Triple


class HDF5KnowledgeGraphLedger(KnowledgeGraphLedger[HDF5KGLedgerConfig]):
    def __init__(self, config: HDF5KGLedgerConfig) -> None:
        super().__init__(config)

        self._initialize_file()

    def _initialize_file(self):
        cache_dir = self.config.file_path.parent
        if not cache_dir.exists():
            logger.info(
                f"Path `{self.config.file_path}` is given as cache file path, "
                f"but `{cache_dir}` does not exist. Newly create `{cache_dir}`."
            )
            cache_dir.mkdir(parents=True)
        with h5py.File(self.config.file_path, "a") as f:
            if "concepts" not in f:
                f.create_group("concepts")
            if "triples" not in f:
                f.create_group("triples")

    def get_concepts_from_domain(self, domain_identifier: DomainIdentifier) -> list[Concept]:
        with h5py.File(self.config.file_path, "r") as f:
            concepts_group = f["concepts"]
            domain_group = concepts_group.get(domain_identifier)

            if domain_group is None:
                raise KeyError(f"Domain group {domain_identifier} not found")

            if not domain_group.attrs.get("is_complete"):
                raise KeyError(f"Domain group {domain_identifier} is found, but is not complete")

            domain_concepts = []
            for concept_identifier in domain_group:
                concept_data = domain_group[concept_identifier]
                concept = Concept(
                    identifier=concept_identifier,
                    label=concept_data.attrs["label"],
                    domain_identifier=domain_identifier,
                )
                domain_concepts.append(concept)

        return domain_concepts

    def add_concept(self, concept: Concept) -> None:
        with h5py.File(self.config.file_path, "a") as f:
            concepts_group = f["concepts"]
            domain_group = concepts_group.require_group(concept.domain_identifier)
            concept_data = domain_group.require_group(concept.identifier)
            concept_data.attrs["label"] = concept.label

    def add_concepts_for_domain(
        self, concepts: Sequence[Concept], domain_identifier: DomainIdentifier
    ):
        with h5py.File(self.config.file_path, "a") as f:
            concepts_group = f["concepts"]
            domain_group = concepts_group.require_group(domain_identifier)
            for concept in concepts:
                if domain_identifier != concept.domain_identifier:
                    raise ValueError(
                        "Domain identifiers do not match, "
                        f"concept identifier: {concept.identifier}, "
                        f"concept domain identifier: {concept.domain_identifier}, "
                        f"domain_identifier given: {domain_identifier}"
                    )
                concept_data = domain_group.require_group(str(concept.identifier))
                concept_data.attrs["label"] = concept.label
            domain_group.attrs["is_complete"] = True

    def _get_concept(self, concept_identifier: ConceptIdentifier) -> Concept:
        with h5py.File(self.config.file_path, "r") as f:
            concepts_group = f["concepts"]
            for domain_name in concepts_group:
                domain_group = concepts_group[domain_name]

                if concept_identifier in domain_group:
                    concept_data = domain_group[concept_identifier]
                    return Concept(
                        identifier=concept_identifier,
                        label=concept_data.attrs["label"],
                        domain_identifier=domain_name,
                    )
        raise KeyError(f"Concept with id {concept_identifier} not found")

    def add_triple(self, triple: Triple):
        raise NotImplementedError

    def _get_triples(
        self,
        subject_identifier: ConceptIdentifier | None = None,
        object_identifier: ConceptIdentifier | None = None,
        relation_identifier: RelationIdentifier | None = None,
    ) -> list[Triple]:
        raise NotImplementedError
