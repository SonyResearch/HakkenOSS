import json

from langchain_core.documents import Document
from psycopg import rows as pg_rows

from hakken_agents.db.config import ColumnInfo, PostgresDBConfig, SQLTableConfig
from hakken_agents.db.engine import PostgresTable


class FactsTable(PostgresTable):
    def __init__(self, db_config: PostgresDBConfig, table_name: str) -> None:
        table_config = SQLTableConfig(
            name=table_name,
            schema_name="public",
            columns=[
                ColumnInfo(name="id", dtype="BIGSERIAL PRIMARY KEY"),
                ColumnInfo(name="subject_uuid", dtype="UUID NOT NULL"),
                ColumnInfo(name="relation_uuid", dtype="UUID NOT NULL"),
                ColumnInfo(name="object_uuid", dtype="UUID NOT NULL"),
                ColumnInfo(name="chunk_uuid", dtype="UUID"),
                ColumnInfo(name="metadata", dtype="JSONB DEFAULT '{}'"),
            ],
            constraints=[
                "UNIQUE (subject_uuid, relation_uuid, object_uuid, chunk_uuid)",
            ],
        )
        super().__init__(db_config, table_config)

    def insert_fact(self, fact_doc: Document) -> int | None:
        """Insert a resolved fact Document into the facts table.

        Expects metadata keys: subject_id, relation_id, object_id, chunk_uuid,
        and optionally subject_quantity / object_quantity.

        Duplicate ``(subject_uuid, relation_uuid, object_uuid, chunk_uuid)``
        tuples are silently skipped via ``ON CONFLICT DO NOTHING``.
        """
        extra: dict = {}
        if fact_doc.metadata.get("subject_quantity"):
            extra["subject_quantity"] = fact_doc.metadata["subject_quantity"]
        if fact_doc.metadata.get("object_quantity"):
            extra["object_quantity"] = fact_doc.metadata["object_quantity"]

        data = {
            "subject_uuid": fact_doc.metadata["subject_id"],
            "relation_uuid": fact_doc.metadata["relation_id"],
            "object_uuid": fact_doc.metadata["object_id"],
            "chunk_uuid": fact_doc.metadata.get("chunk_uuid"),
            "metadata": json.dumps(extra),
        }

        columns = list(data.keys())
        cols_str = ", ".join(columns)
        placeholders = ", ".join(f"%({k})s" for k in columns)

        sql = f"""
        INSERT INTO {self.table_name} ({cols_str})
        VALUES ({placeholders})
        ON CONFLICT (subject_uuid, relation_uuid, object_uuid, chunk_uuid)
        DO NOTHING
        RETURNING id;
        """

        with self.pool.connection() as conn:
            with conn.cursor(row_factory=pg_rows.dict_row) as cur:
                cur.execute(sql, data)
                row = cur.fetchone()
            conn.commit()

        return row["id"] if row else None
