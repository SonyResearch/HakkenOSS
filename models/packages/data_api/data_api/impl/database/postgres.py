from typing import TYPE_CHECKING

import psycopg
from loguru import logger
from psycopg.rows import DictRow, dict_row

from data_api.entities.config import PostgresDatabaseConfig

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import LiteralString

from abc import ABC
from typing import TYPE_CHECKING, Generic, TypeVar

T = TypeVar("T")


_GET_NODENAMES_QUERY: "LiteralString" = """\
    SELECT id, name
    FROM node
    WHERE id = ANY(%s)
"""

_GET_UNIQUEDOMAINNAMES_QUERY: "LiteralString" = """\
    SELECT DISTINCT(name)
    FROM DOMAIN
"""

_GET_NODESFROMDOMAIN_QUERY: "LiteralString" = """\
    SELECT node_id, node_name
    FROM node_domain
    WHERE domain_name = %s
    LIMIT %s
"""

_GET_NODESFROMDOMAINWITHNAME_QUERY: "LiteralString" = """\
    SELECT node_id, node_name
    FROM node_domain
    WHERE domain_name = %s AND LOWER(node_name) LIKE %s
    LIMIT %s
"""

_GET_EDGETYPE_WITHOBJECTDOMAIN_QUERY: "LiteralString" = """\
    SELECT DISTINCT(relation_type)
    FROM public.domain_relation_domain
    WHERE object_domain_name = %s
"""

_GET_EDGETYPE_WITHSUBJECTDOMAIN_QUERY: "LiteralString" = """\
    SELECT DISTINCT(relation_type)
    FROM public.domain_relation_domain
    WHERE subject_domain_name = %s
"""

_GET_EDGETYPE_WITHBOTHDOMAINS_QUERY: "LiteralString" = """\
    SELECT DISTINCT(relation_type)
    FROM public.domain_relation_domain
    WHERE subject_domain_name = %s AND object_domain_name = %s
"""

_GET_EDGETYPE_WITHNODOMAINS_QUERY: "LiteralString" = """\
    SELECT DISTINCT(relation_type)
    FROM public.domain_relation_domain
"""

_GET_NODEDOMAIN_WITHOBJECTDOMAIN_QUERY: "LiteralString" = """\
    SELECT DISTINCT(subject_domain_name) AS domain_names
    FROM public.domain_relation_domain
    WHERE object_domain_name = %s AND relation_type = %s
"""

_GET_NODEDOMAIN_WITHSUBJECTDOMAIN_QUERY: "LiteralString" = """\
    SELECT DISTINCT(object_domain_name) AS domain_names
    FROM public.domain_relation_domain
    WHERE subject_domain_name = %s AND relation_type = %s
"""

_GET_NODEDOMAIN_WITHOBJECTDOMAIN_AND_NOEDGETYPE_QUERY: "LiteralString" = """\
    SELECT DISTINCT(subject_domain_name) AS domain_names
    FROM public.domain_relation_domain
    WHERE object_domain_name = %s
"""

_GET_NODEDOMAIN_WITHSUBJECTDOMAIN_AND_NOEDGETYPE_QUERY: "LiteralString" = """\
    SELECT DISTINCT(object_domain_name) AS domain_names
    FROM public.domain_relation_domain
    WHERE subject_domain_name = %s
"""

_GET_NODEDOMAIN_WITHOBJECTDOMAINEMPTY_AND_EDGETYPE_QUERY: "LiteralString" = """\
    SELECT DISTINCT(subject_domain_name) AS domain_names
    FROM public.domain_relation_domain
    WHERE relation_type = %s
"""

_GET_NODEDOMAIN_WITHSUBJECTDOMAINEMPTY_AND_EDGETYPE_QUERY: "LiteralString" = """\
    SELECT DISTINCT(object_domain_name) AS domain_names
    FROM public.domain_relation_domain
    WHERE relation_type = %s
"""


class ReferenceDatabase(ABC, Generic[T]):
    def __init__(self, config: T) -> None:
        self.config = config


class PostgresDatabase(ReferenceDatabase[PostgresDatabaseConfig]):
    def __init__(self, config: PostgresDatabaseConfig):
        super().__init__(config)
        self._connection: psycopg.Connection[DictRow] | None = None

    def _get_connection(self) -> psycopg.Connection[DictRow]:
        if self._connection is None or self._connection.closed:
            connection_string = (
                f"host={self.config.host} "
                f"port={self.config.port} "
                f"dbname={self.config.db} "
                f"user={self.config.user} "
                f"password={self.config.password}"
            )
            logger.info("Connecting to Postgres database ...")
            connection = psycopg.connect(conninfo=connection_string, row_factory=dict_row)
            self._connection = connection

        return self._connection

    def get_nodenames(self, node_ids: "Sequence[str]") -> dict[str, str]:
        node_ids = list(node_ids)
        connection = self._get_connection()

        logger.info(f"Fetching nodenames for {len(node_ids)} node IDs from Postgres database.")

        with connection.cursor() as cursor:
            cursor.execute(_GET_NODENAMES_QUERY, (node_ids,))
            rows = cursor.fetchall()

        nodename_mapping = {row["id"]: row["name"] for row in rows}
        logger.info(f"Retrieved {len(nodename_mapping)} nodenames from Postgres database.")
        return nodename_mapping

    def get_unique_domains(
        self,
    ) -> list[str]:
        connection = self._get_connection()

        logger.info("Fetching unique domain names.")

        with connection.cursor() as cursor:
            cursor.execute(_GET_UNIQUEDOMAINNAMES_QUERY)
            rows = cursor.fetchall()

        return [row["name"] for row in rows]

    def get_nodes_from_domain(
        self, domain: str, node: str | None = None, max_results: int = 5
    ) -> list[dict[str, str]]:
        connection = self._get_connection()

        logger.info(
            f"Fetching node ids from a given domain {domain} and max results {max_results}."
        )

        with connection.cursor() as cursor:
            if node is None:
                cursor.execute(_GET_NODESFROMDOMAIN_QUERY, (domain, max_results))
            else:
                cursor.execute(
                    _GET_NODESFROMDOMAINWITHNAME_QUERY,
                    (domain, f"%{str.lower(node)}%", max_results),
                )
            rows = cursor.fetchall()

        return [{"id": row["node_id"], "name": row["node_name"]} for row in rows]

    def get_edge_types_from_node_domains(
        self, subject: str | None, object: str | None
    ) -> list[str]:
        connection = self._get_connection()

        logger.info(
            f"Fetching edge types from node subject domain: {subject}, \
                and object domain: {object}."
        )

        with connection.cursor() as cursor:
            if subject is None and object is None:
                cursor.execute(_GET_EDGETYPE_WITHNODOMAINS_QUERY)
            elif subject is None:
                cursor.execute(_GET_EDGETYPE_WITHOBJECTDOMAIN_QUERY, (f"{object}",))
            elif object is None:
                cursor.execute(_GET_EDGETYPE_WITHSUBJECTDOMAIN_QUERY, (f"{subject}",))
            else:
                cursor.execute(
                    _GET_EDGETYPE_WITHBOTHDOMAINS_QUERY,
                    (f"{subject}", f"{object}"),
                )
            rows = cursor.fetchall()

        return [row["relation_type"] for row in rows]

    def get_node_domains_from_node_domain_and_edge_type(
        self, subject: str | None, object: str | None, edge: str | None
    ) -> list[str]:
        connection = self._get_connection()

        logger.info(
            f"Fetching node domains from subject domain: {subject}, \
            object domain: {object}, and edge type: {edge}."
        )
        logger.info(f"object is none: {object is None}")

        with connection.cursor() as cursor:
            if edge is not None:
                logger.info("0")
                if subject is None and object == "":
                    logger.info("1")
                    cursor.execute(
                        _GET_NODEDOMAIN_WITHOBJECTDOMAINEMPTY_AND_EDGETYPE_QUERY, (f"{edge}",)
                    )
                elif object is None and subject == "":
                    logger.info("2")
                    cursor.execute(
                        _GET_NODEDOMAIN_WITHSUBJECTDOMAINEMPTY_AND_EDGETYPE_QUERY, (f"{edge}",)
                    )
                elif subject is None:
                    logger.info("3")
                    cursor.execute(
                        _GET_NODEDOMAIN_WITHOBJECTDOMAIN_QUERY, (f"{object}", f"{edge}")
                    )
                elif object is None:
                    logger.info("4")
                    cursor.execute(
                        _GET_NODEDOMAIN_WITHSUBJECTDOMAIN_QUERY,
                        (f"{subject}", f"{edge}"),
                    )
            elif subject is None:
                logger.info("5")
                cursor.execute(
                    _GET_NODEDOMAIN_WITHOBJECTDOMAIN_AND_NOEDGETYPE_QUERY, (f"{object}",)
                )
            elif object is None:
                logger.info("6")
                cursor.execute(
                    _GET_NODEDOMAIN_WITHSUBJECTDOMAIN_AND_NOEDGETYPE_QUERY, (f"{subject}",)
                )

            rows = cursor.fetchall()

        return [row["domain_names"] for row in rows]
