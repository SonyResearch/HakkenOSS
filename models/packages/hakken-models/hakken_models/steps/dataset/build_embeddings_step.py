"""ZenML steps to build pre-computed embedding matrices for SeGAL.

Both node and relation embeddings are fetched from Postgres pgvector tables
populated by the Element Resolver ingest pipeline.
"""

import json
from typing import Annotated

import numpy as np
import polars as pl
from loguru import logger
from sqlalchemy import create_engine, text
from zenml import ArtifactConfig, log_metadata, step

# ── DB helpers ───────────────────────────────────────────────────────────────


def _fetch_embeddings_from_pgvector(
    pg_connection_string: str,
    table_name: str,
    id_column: str,
) -> dict[str, list[float]]:
    """Query a pgvector table and return ``{id: embedding_vector}``."""
    engine = create_engine(pg_connection_string)

    query = text(f'SELECT "{id_column}", embedding::text FROM "public"."{table_name}"')

    id_to_embedding: dict[str, list[float]] = {}
    with engine.connect() as conn:
        for row in conn.execute(query):
            entity_id = str(row[0])
            id_to_embedding[entity_id] = json.loads(row[1])

    engine.dispose()
    logger.info(f"Fetched {len(id_to_embedding)} embeddings from {table_name}")
    return id_to_embedding


# ── matrix builder ───────────────────────────────────────────────────────────


def _build_ordered_matrix(
    mapping_df: pl.DataFrame,
    id_to_embedding: dict[str, list[float]],
    entity_type: str,
) -> np.ndarray:
    """Build ``[num_entities, D]`` matrix ordered by the mapping's ``index`` column.

    Raises:
        RuntimeError: If any entity in the mapping is missing from *id_to_embedding*.
    """
    sorted_df = mapping_df.sort("index")
    ids: list[str] = sorted_df["id"].cast(pl.Utf8).to_list()

    missing = [eid for eid in ids if eid not in id_to_embedding]
    if missing:
        raise RuntimeError(
            f"{len(missing)} {entity_type}(s) in the mapping have no embedding. "
            f"First 5: {missing[:5]}"
        )

    embedding_dim = len(id_to_embedding[ids[0]])
    matrix = np.empty((len(ids), embedding_dim), dtype=np.float32)
    for i, eid in enumerate(ids):
        matrix[i] = id_to_embedding[eid]

    return matrix


# ── ZenML steps ──────────────────────────────────────────────────────────────


@step
def build_node_embeddings_step(
    nodes_map_df: pl.DataFrame,
    vector_table_name: str,
    pg_connection_string: str,
) -> Annotated[np.ndarray, ArtifactConfig(name="node_embeddings_np")]:
    """Fetch pre-computed node embeddings from Postgres pgvector.

    Queries the given vector table, joins with the node mapping DataFrame
    on ``node_id``, and produces a ``[num_nodes, embedding_dim]`` matrix
    ordered by node index.
    """
    logger.info(f"Fetching node embeddings from table: {vector_table_name}")

    id_to_embedding = _fetch_embeddings_from_pgvector(
        pg_connection_string=pg_connection_string,
        table_name=vector_table_name,
        id_column="node_id",
    )

    matrix = _build_ordered_matrix(
        mapping_df=nodes_map_df,
        id_to_embedding=id_to_embedding,
        entity_type="node",
    )

    log_metadata(
        metadata={
            "shape": list(matrix.shape),
            "dtype": str(matrix.dtype),
            "num_nodes": nodes_map_df.height,
            "embedding_dim": matrix.shape[1],
            "source_table": vector_table_name,
            "db_embeddings_count": len(id_to_embedding),
        },
    )

    logger.info(f"✅ Built node embedding matrix: {matrix.shape}")
    return matrix


@step
def build_relation_embeddings_step(
    relations_map_df: pl.DataFrame,
    vector_table_name: str,
    pg_connection_string: str,
    id_column: str = "relation_id",
) -> Annotated[np.ndarray, ArtifactConfig(name="relation_embeddings_np")]:
    """Fetch pre-computed relation embeddings from Postgres pgvector.

    Queries the given vector table, joins with the relation mapping DataFrame
    on *id_column*, and produces a ``[num_relations, embedding_dim]`` matrix
    ordered by relation index.
    """
    logger.info(f"Fetching relation embeddings from table: {vector_table_name}")

    id_to_embedding = _fetch_embeddings_from_pgvector(
        pg_connection_string=pg_connection_string,
        table_name=vector_table_name,
        id_column=id_column,
    )

    matrix = _build_ordered_matrix(
        mapping_df=relations_map_df,
        id_to_embedding=id_to_embedding,
        entity_type="relation",
    )

    log_metadata(
        metadata={
            "shape": list(matrix.shape),
            "dtype": str(matrix.dtype),
            "num_relations": relations_map_df.height,
            "embedding_dim": matrix.shape[1],
            "source_table": vector_table_name,
            "db_embeddings_count": len(id_to_embedding),
        },
    )

    logger.info(f"✅ Built relation embedding matrix: {matrix.shape}")
    return matrix
