import sqlite3
from typing import TYPE_CHECKING

from loguru import logger
from query_common.entities.kg.concept import Concept
from query_common.entities.kg.identifier import DomainIdentifier

from complex_query.core.contracts import KnowledgeGraphLedger
from complex_query.core.entities.config.kg_ledger import SqliteKGLedgerConfig

if TYPE_CHECKING:
    from collections.abc import Sequence

    from query_common.entities.kg.identifier import (
        ConceptIdentifier,
        DomainIdentifier,
        RelationIdentifier,
    )
    from query_common.entities.kg.triple import Triple


class SqliteKnowledgeGraphLedger(KnowledgeGraphLedger[SqliteKGLedgerConfig]):
    def __init__(self, config: SqliteKGLedgerConfig) -> None:
        super().__init__(config)

        self._initialize_db()

    def _initialize_db(self):
        db_dir = self.config.file_path.parent
        if not db_dir.exists():
            logger.info(
                f"Path `{self.config.file_path}` is given as SQLite file path, "
                f"but `{db_dir}` does not exist. Newly create `{db_dir}`."
            )
            db_dir.mkdir(parents=True)

        with sqlite3.connect(self.config.file_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                "CREATE TABLE IF NOT EXISTS kg_cache ("
                "   id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "   concept_identifier TEXT UNIQUE NOT NULL,"
                "   domain_identifier TEXT NOT NULL,"
                "   concept_label TEXT NOT NULL"
                ")"
            )
            cursor.execute(
                "CREATE TABLE IF NOT EXISTS complete_domain ("
                "   id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "   domain_identifier TEXT UNIQUE NOT NULL"
                ")"
            )

            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_concept_identifier ON kg_cache(concept_identifier)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_domain_identifier ON kg_cache(domain_identifier)"
            )

            conn.commit()

    def add_concept(self, concept: "Concept") -> None:
        query_params = {
            "concept_identifier": concept.identifier,
            "domain_identifier": concept.domain_identifier,
            "concept_label": concept.label,
        }

        with sqlite3.connect(self.config.file_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                (
                    "INSERT OR IGNORE INTO kg_cache "
                    "   (concept_identifier, domain_identifier, concept_label) "
                    "VALUES (:concept_identifier, :domain_identifier, :concept_label)"
                ),
                query_params,
            )
            conn.commit()

    def add_concepts_for_domain(
        self, concepts: "Sequence[Concept]", domain_identifier: "DomainIdentifier"
    ) -> None:
        query_params_list = []
        for concept in concepts:
            if domain_identifier != concept.domain_identifier:
                raise ValueError(
                    "Domain identifiers do not match, "
                    f"concept identifier: {concept.identifier}, "
                    f"concept domain identifier: {concept.domain_identifier}, "
                    f"domain_identifier given: {domain_identifier}"
                )
            query_params_list.append(
                {
                    "concept_identifier": concept.identifier,
                    "domain_identifier": concept.domain_identifier,
                    "concept_label": concept.label,
                }
            )

        with sqlite3.connect(self.config.file_path) as conn:
            cursor = conn.cursor()
            cursor.executemany(
                (
                    "INSERT OR IGNORE INTO kg_cache "
                    "   (concept_identifier, domain_identifier, concept_label) "
                    "VALUES (:concept_identifier, :domain_identifier, :concept_label)"
                ),
                query_params_list,
            )
            cursor.execute(
                "INSERT INTO complete_domain (domain_identifier) VALUES (:domain_identifier)",
                {"domain_identifier": domain_identifier},
            )
            conn.commit()

    def _get_concept(self, concept_identifier: "ConceptIdentifier") -> "Concept":
        with sqlite3.connect(self.config.file_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                (
                    "SELECT concept_identifier, domain_identifier, concept_label "
                    "FROM kg_cache "
                    "WHERE concept_identifier = :concept_identifier"
                ),
                {"concept_identifier": concept_identifier},
            )

            row = cursor.fetchone()
            if not row:
                raise KeyError(f"Concept with id {concept_identifier} not found")

            return Concept(
                identifier=row["concept_identifier"],
                label=row["concept_label"],
                domain_identifier=row["domain_identifier"],
            )

    def get_concepts_from_domain(self, domain_identifier: str) -> list[Concept]:
        with sqlite3.connect(self.config.file_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                (
                    "SELECT domain_identifier FROM complete_domain "
                    "WHERE domain_identifier = :domain_identifier"
                ),
                {"domain_identifier": domain_identifier},
            )
            row = cursor.fetchone()
            if not row:
                raise KeyError(f"Domain {domain_identifier} is not available in ledger")

            cursor.execute(
                (
                    "SELECT concept_identifier, domain_identifier, concept_label "
                    "FROM kg_cache "
                    "WHERE domain_identifier = :domain_identifier"
                ),
                {"domain_identifier": domain_identifier},
            )
            rows = cursor.fetchall()
            return [
                Concept(
                    identifier=row["concept_identifier"],
                    label=row["concept_label"],
                    domain_identifier=row["domain_identifier"],
                )
                for row in rows
            ]

    def add_triple(self, triple: "Triple") -> None:
        raise NotImplementedError

    def _get_triples(
        self,
        subject_identifier: "ConceptIdentifier | None" = None,
        object_identifier: "ConceptIdentifier | None" = None,
        relation_identifier: "RelationIdentifier | None" = None,
    ) -> list["Triple"]:
        raise NotImplementedError
