from typing import Self

from pydantic import Field, model_validator

from .base_settings import HakkenSettings


class DatasetPreparatorConfig(HakkenSettings):
    dataset_name: str = Field(..., description="e.g. pubtator3-v0.4.0")
    dataset_version: str = Field(..., description="e.g. v17")

    s3_raw_dir: str = "s3://data/{dataset_name}"

    output_dir: str | None = "s3://data/{dataset_name}/{dataset_version}"

    allowed_relations: list[str] | None = Field(None)

    temporal_partitions: dict[str, tuple[str | None, str | None]] | None = Field(
        default=None,
        description=(
            "Defines temporal partitions for train/val/test splits based on date ranges. "
            "Keys are partition names (e.g., 'train', 'val', 'test'). "
            "Values are tuples of (start_date, end_date) where both dates are optional string ids. "
            "To filter datasets by date column: rows with date >= start_date and date < end_date. "
            "Example: {'train': ('2020-01-01', '2022-12-31'), 'val': ('2023-01-01', '2023-06-30')}"
        ),
    )

    # ── Postgres vector DB (for pre-computed embedding fetch) ─────────
    pg_host: str = Field(default="localhost", description="Postgres host for embedding vectors")
    pg_port: int = Field(default=5432, description="Postgres port for embedding vectors")
    pg_user: str = Field(default="postgres", description="Postgres user")
    pg_password: str = Field(default="postgres", description="Postgres password")
    pg_database: str = Field(default="hakken_agents", description="Postgres database name")

    @property
    def pg_connection_string(self) -> str:
        """SQLAlchemy connection string for the embedding vector DB."""
        return (
            f"postgresql://{self.pg_user}:{self.pg_password}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_database}"
        )

    @property
    def _sanitized_dataset_name(self) -> str:
        return self.dataset_name.replace(".", "_")

    @property
    def node_vectors_table_name(self) -> str:
        """Derive the node pgvector table name from the dataset name.

        Convention: dots replaced with underscores, suffixed with ``-nodes_vectors``.
        E.g. ``pubtator3-v0.4.0`` → ``pubtator3-v0_4_0-nodes_vectors``.
        """
        return f"{self._sanitized_dataset_name}-nodes_vectors"

    @property
    def relation_vectors_table_name(self) -> str:
        """Derive the relation pgvector table name from the dataset name.

        Convention: dots replaced with underscores, suffixed with ``-relations_vectors``.
        E.g. ``pubtator3-v0.4.0`` → ``pubtator3-v0_4_0-relations_vectors``.
        """
        return f"{self._sanitized_dataset_name}-relations_vectors"

    @model_validator(mode="after")
    def fill_s3_paths(self) -> Self:
        """Replace placeholders in S3 paths using dataset info."""
        self.s3_raw_dir = self.s3_raw_dir.format(dataset_name=self.dataset_name)

        if self.output_dir is not None:
            self.output_dir = self.output_dir.format(
                dataset_name=self.dataset_name,
                dataset_version=self.dataset_version,
            )
        return self
