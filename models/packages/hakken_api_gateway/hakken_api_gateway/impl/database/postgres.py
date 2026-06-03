import json
import uuid
from abc import ABC
from datetime import datetime
from typing import TYPE_CHECKING, Generic, TypeVar

import psycopg
from loguru import logger
from psycopg import sql
from psycopg.rows import DictRow, dict_row
from psycopg.types.json import Jsonb

from hakken_api_gateway.core.entities.config import PostgresDatabaseConfig

if TYPE_CHECKING:
    from hakken_api_gateway.core.entities.user import UserDBModel

T = TypeVar("T")


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
            logger.info("Connecting to Postgres database ....")
            connection = psycopg.connect(conninfo=connection_string, row_factory=dict_row)
            self._connection = connection

        return self._connection

    def get_user(self, email: str, table: str = "user") -> dict | None:
        connection = self._get_connection()

        logger.info(f"Fetching user with email {email} from Postgres database.")

        with connection.cursor() as cursor:
            cursor.execute(
                sql.SQL("SELECT * FROM {} WHERE email = %s").format(sql.Identifier(table)),
                (email,),
            )
            row = cursor.fetchone()

        if row:
            logger.info(f"User with email {email} found in Postgres database.")
            return row
        logger.info(f"No user with email {email} found in Postgres database.")
        return None

    def add_user(self, user_data: "UserDBModel", table: str = "user") -> str | None:
        connection = self._get_connection()

        logger.info(f"Adding new user with email {user_data.email} to Postgres database.")

        user_data.id = str(uuid.uuid4()).replace("-", "")
        now = datetime.now()
        with connection.cursor() as cursor:
            cursor.execute(
                sql.SQL(
                    'INSERT INTO {} (id, email, name, jwt, "createdAt", "updatedAt", "lastLogin") '
                    "VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id"
                ).format(sql.Identifier(table)),
                (
                    user_data.id,
                    user_data.email,
                    user_data.name,
                    user_data.name,
                    now,
                    now,
                    now,
                ),
            )
            record = cursor.fetchone()
            user_id = record["id"] if isinstance(record, dict) else None
            connection.commit()

        logger.info("New user added to Postgres database.")
        return user_id

    def add_query(self, query_data: dict, table: str = "query") -> str | None:
        connection = self._get_connection()

        logger.info("Adding new query to Postgres database.")

        logger.info(f"Generated UUID {query_data['id']} for new query.")
        logger.info(f"len query id: {len(query_data['id'])}")
        now = datetime.now()

        # Parse JSON strings to dicts if needed
        query_field = query_data.get("query")
        response_field = query_data.get("response")

        # If they're JSON strings, parse them; if already dicts, use as-is
        if isinstance(query_field, str):
            query_field = json.loads(query_field)
        if isinstance(response_field, str):
            response_field = json.loads(response_field)

        with connection.cursor() as cursor:
            cursor.execute(
                sql.SQL(
                    'INSERT INTO {} (id, user_id, query, response, "createdAt") '
                    "VALUES (%s, %s, %s, %s, %s) RETURNING id"
                ).format(sql.Identifier(table)),
                (
                    query_data.get("id"),
                    query_data.get("user_id"),
                    Jsonb(query_field),
                    Jsonb(response_field),
                    now,
                ),
            )
            record = cursor.fetchone()
            query_id = record["id"] if isinstance(record, dict) else None
            connection.commit()

        logger.info(f"New query added with ID {query_id} to Postgres database.")
        return query_id

    def get_explanation(self, id: str, table: str = "explanation") -> dict | None:
        connection = self._get_connection()

        logger.info(f"Fetching explanation with ID {id} from Postgres database.")

        with connection.cursor() as cursor:
            cursor.execute(
                sql.SQL("SELECT * FROM {} WHERE id = %s").format(sql.Identifier(table)),
                (id,),
            )
            row = cursor.fetchone()

        if row:
            logger.info(f"Explanation with ID {id} found in Postgres database.")
            return row
        logger.info(f"No explanation with ID {id} found in Postgres database.")
        return None

    def add_explanation(self, explanation_data: dict, table: str = "explanation") -> str | None:
        connection = self._get_connection()

        logger.info("Adding new explanation to Postgres database.")

        logger.info(f"Generated UUID {explanation_data['id']} for new explanation.")
        logger.info(f"len explanation id: {len(explanation_data['id'])}")
        now = datetime.now()
        with connection.cursor() as cursor:
            cursor.execute(
                sql.SQL(
                    'INSERT INTO {} (id, query, explanations, "createdAt", "updatedAt") '
                    "VALUES (%s, %s, %s, %s, %s) RETURNING id"
                ).format(sql.Identifier(table)),
                (
                    explanation_data.get("id"),
                    Jsonb(explanation_data.get("query")),
                    Jsonb(explanation_data.get("explanations")),
                    now,
                    now,
                ),
            )
            record = cursor.fetchone()
            explanation_id = record["id"] if isinstance(record, dict) else None
            connection.commit()

        logger.info(f"New explanation added with ID {explanation_id} to Postgres database.")
        return explanation_id

    def get_query(self, id: str, table: str = "query") -> dict | None:
        connection = self._get_connection()

        logger.info(f"Fetching query with ID {id} from Postgres database.")

        with connection.cursor() as cursor:
            cursor.execute(
                sql.SQL("SELECT * FROM {} WHERE id = %s").format(sql.Identifier(table)),
                (id,),
            )
            row = cursor.fetchone()

        if row:
            logger.info(f"Query with ID {id} found in Postgres database.")
            return row
        logger.info(f"No query with ID {id} found in Postgres database.")
        return None

    def get_user_queries(self, id: str, table: str = "query") -> list | None:
        connection = self._get_connection()

        logger.info("Fetching queries for user from Postgres database.")

        with connection.cursor() as cursor:
            cursor.execute(
                sql.SQL("SELECT * FROM {} WHERE user_id = %s").format(sql.Identifier(table)),
                (id,),
            )
            row = cursor.fetchall()

        if row:
            logger.info("Queries for user found.")
            return row
        logger.info("No query for user found.")
        return None

    def delete_query(self, query_id: str, user_id: str, table: str = "query") -> bool:
        connection = self._get_connection()

        logger.info(f"Deleting query with ID {query_id} from Postgres database.")

        with connection.cursor() as cursor:
            cursor.execute(
                sql.SQL("DELETE FROM {} WHERE id = %s AND user_id = %s RETURNING id").format(
                    sql.Identifier(table)
                ),
                (query_id, user_id),
            )
            row = cursor.fetchone()
            connection.commit()

        if row:
            logger.info(f"Query with ID {query_id} deleted from Postgres database.")
            return True
        logger.info(f"No query with ID {query_id} found for user.")
        return False

    def delete_all_user_queries(self, user_id: str, table: str = "query") -> bool:
        connection = self._get_connection()

        logger.info("Deleting all queries for user from Postgres database.")

        with connection.cursor() as cursor:
            cursor.execute(
                sql.SQL("DELETE FROM {} WHERE user_id = %s RETURNING id").format(
                    sql.Identifier(table)
                ),
                (user_id,),
            )
            rows = cursor.fetchall()
            connection.commit()

        if rows:
            logger.info("All queries for user deleted from Postgres database.")
            return True
        logger.info("No queries found for user in Postgres database.")
        return False
