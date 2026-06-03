from collections import defaultdict
from typing import TYPE_CHECKING

import psycopg
from psycopg import sql
from psycopg.rows import DictRow, dict_row

from contextualization.core.contracts.reference_database import ReferenceDatabase
from contextualization.core.entities.config.reference_database import (
    PostgresReferenceDatabaseConfig,
)
from contextualization.core.entities.link import ConceptId, PublicationConceptLink
from contextualization.core.entities.publication import Publication, PublicationId
from contextualization.core.values.errors import (
    PublicationNotFoundError,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import LiteralString

_GET_PUBLICATIONS_QUERY: "LiteralString" = """\
    SELECT publication_id, year, title, abstract, doi, citations_count, authors
    FROM {table_name}
    WHERE publication_id = ANY(%(publication_ids)s)
"""
_GET_PUBLICATION_CONCEPT_LINKS_FROM_PUBLICATION_IDS_QUERY: "LiteralString" = """\
    SELECT publication_id, concept_id
    FROM {publication_table_name} pub JOIN {publication_concept_table_name} pub_concept
        USING (publication_id)
    WHERE publication_id = ANY(%(publication_ids)s)
    ORDER BY citations_count DESC, year DESC
"""
_GET_PUBLICATION_CONCEPT_LINKS_FROM_PUBLICATION_IDS_WITH_MAX_SIZE_QUERY: "LiteralString" = """\
    WITH ranked_concept_links AS (
        SELECT publication_id, concept_id,
            ROW_NUMBER() OVER (
                PARTITION BY publication_id
                ORDER BY citations_count DESC, year DESC
            ) AS rn
            FROM {publication_table_name} pub JOIN {publication_concept_table_name} pub_concept
                USING (publication_id)
            WHERE publication_id = ANY(%(publication_ids)s)
    )
    SELECT publication_id, concept_id
    FROM ranked_concept_links
    WHERE rn <= %(max_size)s
    ORDER BY publication_id, rn
"""
_GET_PUBLICATION_CONCEPT_LINKS_FROM_CONCEPT_IDS_QUERY: "LiteralString" = """\
    SELECT publication_id, concept_id
    FROM {publication_table_name} pub JOIN {publication_concept_table_name} pub_concept
        USING (publication_id)
    WHERE concept_id = ANY(%(concept_ids)s)
    ORDER BY citations_count DESC, year DESC
"""
_GET_PUBLICATION_CONCEPT_LINKS_FROM_CONCEPT_IDS_WITH_MAX_SIZE_QUERY: "LiteralString" = """\
    WITH ranked_concept_links AS (
        SELECT publication_id, concept_id,
            ROW_NUMBER() OVER (
                PARTITION BY concept_id
                ORDER BY citations_count DESC, year DESC
            ) AS rn
            FROM {publication_table_name} pub JOIN {publication_concept_table_name} pub_concept
                USING (publication_id)
            WHERE concept_id = ANY(%(concept_ids)s)
    )
    SELECT publication_id, concept_id
    FROM ranked_concept_links
    WHERE rn <= %(max_size)s
    ORDER BY concept_id, rn
"""


class PostgresReferenceDatabase(ReferenceDatabase[PostgresReferenceDatabaseConfig]):
    def __init__(self, config: PostgresReferenceDatabaseConfig):
        super().__init__(config)

        self._connection: psycopg.Connection[DictRow] | None = None

    def _get_connection(self) -> psycopg.Connection[DictRow]:
        if self._connection is None or self._connection.closed:
            connection = psycopg.connect(
                conninfo=self.config.connection_string, row_factory=dict_row
            )
            self._connection = connection

        return self._connection

    def get_publications(self, publication_ids: "Sequence[PublicationId]") -> list[Publication]:
        publication_ids = list(publication_ids)
        connection = self._get_connection()

        with connection.cursor() as cursor:
            cursor.execute(
                sql.SQL(_GET_PUBLICATIONS_QUERY).format(
                    table_name=sql.Identifier(self.config.publication_table_name)
                ),
                params={"publication_ids": publication_ids},
            )
            rows = cursor.fetchall()

        publications_unordered = [Publication(**row) for row in rows]

        retrieved_publication_ids = {
            publication.publication_id for publication in publications_unordered
        }
        for publication_id in publication_ids:
            if publication_id not in retrieved_publication_ids:
                raise PublicationNotFoundError(publication_id=publication_id)

        publication_id_to_index = {
            publication_id: i for i, publication_id in enumerate(publication_ids)
        }
        indices_for_ordering = [
            publication_id_to_index[publication.publication_id]
            for publication in publications_unordered
        ]

        return [publications_unordered[i] for i in indices_for_ordering]

    def _get_publication_concept_links_from_publication_ids(
        self,
        publication_ids: "Sequence[PublicationId]",
        per_publication_max_size: int | None = None,
    ) -> list[list[PublicationConceptLink]]:
        publication_ids = list(publication_ids)
        connection = self._get_connection()

        with connection.cursor() as cursor:
            if per_publication_max_size:
                cursor.execute(
                    sql.SQL(
                        _GET_PUBLICATION_CONCEPT_LINKS_FROM_PUBLICATION_IDS_WITH_MAX_SIZE_QUERY
                    ).format(
                        publication_table_name=sql.Identifier(self.config.publication_table_name),
                        publication_concept_table_name=sql.Identifier(
                            self.config.publication_concept_link_table_name
                        ),
                    ),
                    params={
                        "publication_ids": publication_ids,
                        "max_size": per_publication_max_size,
                    },
                )
            else:
                cursor.execute(
                    sql.SQL(_GET_PUBLICATION_CONCEPT_LINKS_FROM_PUBLICATION_IDS_QUERY).format(
                        publication_table_name=sql.Identifier(self.config.publication_table_name),
                        publication_concept_table_name=sql.Identifier(
                            self.config.publication_concept_link_table_name
                        ),
                    ),
                    params={"publication_ids": publication_ids},
                )
            rows = cursor.fetchall()

        publication_concept_links_ungrouped = [PublicationConceptLink(**row) for row in rows]
        publication_id_to_publication_concept_links: dict[
            PublicationId, list[PublicationConceptLink]
        ] = defaultdict(list)
        for publication_concept_link in publication_concept_links_ungrouped:
            publication_id_to_publication_concept_links[
                publication_concept_link.publication_id
            ].append(publication_concept_link)

        links_list: list[list[PublicationConceptLink]] = []
        for publication_id in publication_ids:
            links_list.append(publication_id_to_publication_concept_links[publication_id])

        return links_list

    def _get_publication_concept_links_from_concept_ids(
        self, concept_ids: "Sequence[ConceptId]", per_concept_max_size: int | None = None
    ) -> list[list[PublicationConceptLink]]:
        concept_ids = list(concept_ids)
        connection = self._get_connection()

        with connection.cursor() as cursor:
            if per_concept_max_size:
                cursor.execute(
                    sql.SQL(
                        _GET_PUBLICATION_CONCEPT_LINKS_FROM_CONCEPT_IDS_WITH_MAX_SIZE_QUERY
                    ).format(
                        publication_table_name=sql.Identifier(self.config.publication_table_name),
                        publication_concept_table_name=sql.Identifier(
                            self.config.publication_concept_link_table_name
                        ),
                    ),
                    params={"concept_ids": concept_ids, "max_size": per_concept_max_size},
                )
            else:
                cursor.execute(
                    sql.SQL(_GET_PUBLICATION_CONCEPT_LINKS_FROM_CONCEPT_IDS_QUERY).format(
                        publication_table_name=sql.Identifier(self.config.publication_table_name),
                        publication_concept_table_name=sql.Identifier(
                            self.config.publication_concept_link_table_name
                        ),
                    ),
                    params={"concept_ids": concept_ids},
                )
            rows = cursor.fetchall()

        publication_concept_links_ungrouped = [PublicationConceptLink(**row) for row in rows]
        concept_id_to_publication_concept_links: dict[ConceptId, list[PublicationConceptLink]] = (
            defaultdict(list)
        )
        for publication_concept_link in publication_concept_links_ungrouped:
            concept_id_to_publication_concept_links[publication_concept_link.concept_id].append(
                publication_concept_link
            )

        links_list: list[list[PublicationConceptLink]] = []
        for concept_id in concept_ids:
            links_list.append(concept_id_to_publication_concept_links[concept_id])

        return links_list
