from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class IngestSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="INGEST_", extra="ignore")
    NODES_FILE_PATH: str | None = Field(None, description="Absolute path to the nodes file")
    EDGES_FILE_PATH: str | None = Field(None, description="Absolute path to the edges file")
    RECREATE_GRAPH: bool = Field(True, description="Drop existing graph before load")


class QuerySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="QUERY_", extra="ignore")
    FILE: str
    TIMEOUT_MS: int


class TemporalKGSettings(BaseSettings):
    """
    Base settings for Temporal KG engines.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    GRAPH_NAME: str = Field(default="temporal_kg", description="Target graph name")
    BATCH_SIZE: int = Field(default=10_000, ge=1, description="Batch size for bulk ops")
    NODE_PROPERTIES: dict[str, str] | None = Field(
        default=None,
        description="Mapping from dataframe columns to graph property names",
        examples=[{"node_name": "name", "node_domain_id": "domain_id"}],
    )
    LOCAL_DATA_STORAGE_DIR: str = Field(
        default="./data/{graph_name}",
        description=(
            "Local directory path for storing graph data files (backup, facts TSV, nodes TSV). "
            "Supports {graph_name} placeholder."
        ),
    )

    S3_BACKUP_DIR: str = Field(
        default="s3://<your-bucket>/data/processed/data_processing/{graph_name}/backup/",
        description=(
            "S3 backup directory path in format s3://bucket/path/to/backup. "
            "Supports {graph_name} placeholder. Set via INGEST_S3_BACKUP_DIR env var or .env file."
        ),
    )

    S3_FACTS_TSV_PATH: str = Field(
        default="s3://<your-bucket>/data/processed/data_processing/{graph_name}/edges.tsv",
        description=(
            "S3 path to edges/facts TSV file in format s3://bucket/path/to/file.tsv. "
            "Supports {graph_name} placeholder. Set via INGEST_S3_FACTS_TSV_PATH env var or .env file."
        ),
    )

    S3_NODES_TSV_PATH: str = Field(
        default="s3://<your-bucket>/data/processed/data_processing/{graph_name}/nodes_corrected.tsv",
        description=(
            "S3 path to nodes TSV file in format s3://bucket/path/to/file.tsv. "
            "Supports {graph_name} placeholder. Set via INGEST_S3_NODES_TSV_PATH env var or .env file."
        ),
    )
