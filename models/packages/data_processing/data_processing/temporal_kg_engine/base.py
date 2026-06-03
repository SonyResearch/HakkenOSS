from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Generic, TypeVar

import boto3
import polars as pl
from botocore.exceptions import ClientError, NoCredentialsError
from loguru import logger

from data_processing.temporal_kg_engine.exceptions import DataNotLoadedError

from .settings import TemporalKGSettings

T_Settings = TypeVar("T_Settings", bound=TemporalKGSettings)


class TemporalKGEngine(ABC, Generic[T_Settings]):
    """
    Abstract base class for Temporal Knowledge Graph Engines.

    Defines the interface for managing and querying temporal knowledge graphs
    with support for domain-typed nodes and relation-typed edges.
    """

    def __init__(
        self,
        graph_name: str = "temporal_kg",
        batch_size: int = 10000,
        node_properties: dict[str, str] | None = None,
        s3_nodes_tsv_path: str | None = None,
        s3_facts_tsv_path: str | None = None,
        s3_backup_dir: str | None = None,
        local_data_storage_dir: str | None = None,
    ):
        """
        Initialize the Temporal Knowledge Graph Engine.

        Args:
            graph_name: Name of the graph to create/manage
            batch_size: Batch size for bulk operations
        """
        self.graph_name = graph_name
        self.batch_size = batch_size

        self.s3_nodes_tsv_path = s3_nodes_tsv_path
        self.s3_facts_tsv_path = s3_facts_tsv_path
        self.s3_backup_dir = s3_backup_dir
        self.local_data_storage_dir = local_data_storage_dir

        if node_properties is None:
            self.node_properties = {"node_id": "node_id"}
        else:
            self.node_properties = {"node_id": "node_id", **node_properties}

        self._nodes_df: pl.DataFrame | None = None
        self._edges_df: pl.DataFrame | None = None

        self.domain_types: list[str] = []
        self.relation_types: list[str] = []

    @property
    def graph_name_label(self) -> str:
        return self.graph_name.upper()

    @property
    def edges_df(self) -> pl.DataFrame:
        if self._edges_df is None:
            raise DataNotLoadedError()
        return self._edges_df

    @property
    def nodes_df(self) -> pl.DataFrame:
        if self._nodes_df is None:
            raise DataNotLoadedError()
        return self._nodes_df

    @property
    def nodes_tsv_path(self) -> Path | None:
        if self.local_data_storage_dir is None:
            return None
        folder = self.local_data_storage_dir.format(graph_name=self.graph_name)
        basename = self.s3_nodes_tsv_path.split("/")[-1] if self.s3_nodes_tsv_path else "nodes.tsv"
        return Path(folder) / basename

    @property
    def facts_tsv_path(self) -> Path | None:
        if self.local_data_storage_dir is None:
            return None
        folder = self.local_data_storage_dir.format(graph_name=self.graph_name)
        basename = self.s3_facts_tsv_path.split("/")[-1] if self.s3_facts_tsv_path else "edges.tsv"
        return Path(folder) / basename

    @staticmethod
    def _base_args(settings: TemporalKGSettings) -> dict[str, Any]:
        return {
            "graph_name": settings.GRAPH_NAME,
            "batch_size": settings.BATCH_SIZE,
            "node_properties": settings.NODE_PROPERTIES,
            "s3_nodes_tsv_path": settings.S3_NODES_TSV_PATH,
            "s3_facts_tsv_path": settings.S3_FACTS_TSV_PATH,
            "s3_backup_dir": settings.S3_BACKUP_DIR,
            "local_data_storage_dir": settings.LOCAL_DATA_STORAGE_DIR,
        }

    def download_file(self, s3_path: str, local_path: str) -> None:
        s3_client = boto3.client("s3")
        logger.info(f"Downloading file from {s3_path} to {local_path}")
        bucket, key = self._parse_s3_path(s3_path)
        s3_client.download_file(bucket, key, local_path)
        logger.info(f"✅ Successfully downloaded file ({Path(local_path).stat().st_size:,} bytes)")

    def download_data(self, force: bool = False) -> None:  # noqa: PLR0915, PLR0912
        """
        Download nodes and edges data from S3 to local storage directory.

        Args:
            force: If True, overwrite existing local files. If False, skip download
                if files already exist and log a warning.

        Raises:
            ValueError: If S3 paths or local storage directory are not configured
        """
        logger.info("=" * 60)
        logger.info("Downloading Data from S3")
        logger.info("=" * 60)

        # Validate S3 paths are configured
        if not self.s3_nodes_tsv_path:
            msg = "S3 nodes TSV path is not configured"
            raise ValueError(msg)
        if not self.s3_facts_tsv_path:
            msg = "S3 facts TSV path is not configured"
            raise ValueError(msg)
        if not self.local_data_storage_dir:
            msg = "Local data storage directory is not configured"
            raise ValueError(msg)

        # Format paths with graph_name placeholder
        s3_nodes_path = self.s3_nodes_tsv_path.format(graph_name=self.graph_name)
        s3_facts_path = self.s3_facts_tsv_path.format(graph_name=self.graph_name)

        # Get local paths (these properties already handle formatting)
        local_nodes_path = self.nodes_tsv_path
        local_facts_path = self.facts_tsv_path

        if local_nodes_path is None or local_facts_path is None:
            msg = "Could not determine local storage paths"
            raise ValueError(msg)

        # Ensure local directories exist
        local_nodes_path.parent.mkdir(parents=True, exist_ok=True)
        local_facts_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if files already exist when force=False
        if not force:
            nodes_exists = local_nodes_path.exists()
            facts_exists = local_facts_path.exists()

            if nodes_exists and facts_exists:
                logger.info("=" * 60)
                logger.info("✅ Download skipped (files already exist)")
                logger.info("=" * 60)
                return
            if nodes_exists:
                logger.warning(
                    f"⚠️  Nodes file already exists at {local_nodes_path}. "
                    f"Skipping download (use force=True to overwrite)."
                )
                logger.info(f"Facts file missing, will download from {s3_facts_path}")
            elif facts_exists:
                logger.info(f"Nodes file missing, will download from {s3_nodes_path}")
                logger.warning(
                    f"⚠️  Facts file already exists at {local_facts_path}. "
                    f"Skipping download (use force=True to overwrite)."
                )

        # Initialize S3 client

        try:
            # Download nodes file (only if force=True or file doesn't exist)
            should_download_nodes = force or not local_nodes_path.exists()
            if should_download_nodes:
                self.download_file(s3_nodes_path, str(local_nodes_path))
            else:
                logger.info("Skipping nodes download (file already exists)")

            # Download facts file (only if force=True or file doesn't exist)
            should_download_facts = force or not local_facts_path.exists()
            if should_download_facts:
                self.download_file(s3_facts_path, str(local_facts_path))
            else:
                logger.info("Skipping facts download (file already exists)")

            logger.info("=" * 60)
            logger.info("✅ Download Complete")
            logger.info("=" * 60)

        except NoCredentialsError:
            msg = "AWS credentials not found. Please configure AWS credentials."
            logger.error(msg)
            raise
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            msg = f"Failed to download from S3: {error_code} - {e}"
            logger.error(msg)
            raise
        except Exception as e:
            msg = f"Unexpected error during download: {e}"
            logger.error(msg)
            raise

    @staticmethod
    def _parse_s3_path(s3_path: str) -> tuple[str, str]:
        """
        Parse an S3 path into bucket and key components.

        Args:
            s3_path: S3 path in format s3://bucket/path/to/file

        Returns:
            Tuple of (bucket_name, object_key)

        Raises:
            ValueError: If the S3 path format is invalid
        """
        if not s3_path.startswith("s3://"):
            msg = f"Invalid S3 path format: {s3_path}. Must start with 's3://'"
            raise ValueError(msg)

        # Remove 's3://' prefix and split into bucket and key
        path_without_prefix = s3_path[5:]
        parts = path_without_prefix.split("/", 1)

        if len(parts) != 2:
            msg = f"Invalid S3 path format: {s3_path}. Expected format: s3://bucket/key"
            raise ValueError(msg)

        bucket, key = parts
        if not bucket:
            msg = f"Invalid S3 path: bucket name is empty in {s3_path}"
            raise ValueError(msg)

        return bucket, key

    def load_data(
        self,
        nodes_file_path: str | None = None,
        edges_file_path: str | None = None,
        filter_null_years: bool = True,
    ) -> None:
        """
        Load nodes and edges data from TSV files.

        Args:
            nodes_file_path: Path to nodes TSV file
            edges_file_path: Path to edges TSV file
            filter_null_years: Whether to filter out edges with null years
        """
        logger.info("=" * 60)
        logger.info("STEP 1: Loading Data")
        logger.info("=" * 60)
        if nodes_file_path is None:
            nodes_file_path = str(self.nodes_tsv_path)
        if edges_file_path is None:
            edges_file_path = str(self.facts_tsv_path)

        # Load nodes
        self._nodes_df = pl.read_csv(
            nodes_file_path,
            separator="\t",
            schema_overrides={
                "node_id": pl.Utf8,
                "node_domain": pl.Utf8,
                "node_name": pl.Utf8,
                "node_domain_id": pl.Utf8,
            },
        )
        logger.info(f"✅ Loaded {len(self.nodes_df):,} nodes from {nodes_file_path}")

        # Load edges
        self._edges_df = pl.read_csv(
            edges_file_path,
            separator="\t",
            schema_overrides={
                "subject_id": pl.Utf8,
                "subject_domain": pl.Utf8,
                "relation_type": pl.Utf8,
                "object_id": pl.Utf8,
                "object_domain": pl.Utf8,
                "year": pl.Int16,
                "number_of_occurrences": pl.Int32,
            },
        )
        logger.info(f"✅ Loaded {len(self.edges_df):,} edges from {edges_file_path}")

        # Filter out null years
        if filter_null_years:
            edges_df_original = len(self._edges_df)
            self._edges_df = self._edges_df.filter(pl.col("year").is_not_null())
            filtered_count = edges_df_original - len(self._edges_df)
            if filtered_count > 0:
                logger.info(f"⚠️  Filtered out {filtered_count:,} edges with null years")

        logger.info(f"✅ {len(self.edges_df):,} edges ready for ingestion")

        # Extract domain and relation types
        self.domain_types = self.nodes_df["node_domain"].unique().sort().to_list()
        logger.info(f"✅ Found {len(self.domain_types)} domain types: {self.domain_types}")

        self.relation_types = self.edges_df["relation_type"].unique().sort().to_list()
        logger.info(f"✅ Found {len(self.relation_types)} relation types: {self.relation_types}")

    def validate_data(self) -> None:
        """Perform data quality checks."""

        logger.info("=" * 60)
        logger.info("STEP 2: Data Quality Checks")
        logger.info("=" * 60)

        # Check for nulls
        nodes_nulls = self.nodes_df.null_count()
        edges_nulls = self.edges_df.null_count()
        logger.info(f"Null counts in nodes:\n{nodes_nulls}")
        logger.info(f"Null counts in edges:\n{edges_nulls}")

        # Check year range
        year_stats = self.edges_df.select(
            [
                pl.col("year").min().alias("min_year"),
                pl.col("year").max().alias("max_year"),
                pl.col("year").n_unique().alias("unique_years"),
            ]
        )
        logger.info(f"Year range: {year_stats}")

        # Check domain consistency
        edge_domains = set(
            self.edges_df["subject_domain"].unique().to_list()
            + self.edges_df["object_domain"].unique().to_list()
        )
        node_domains = set(self.nodes_df["node_domain"].unique().to_list())

        if edge_domains == node_domains:
            logger.info("✅ All edge domains match node domains")
        else:
            missing = edge_domains - node_domains
            if missing:
                logger.error(f"❌ Edge domains not in nodes: {missing}")
                msg = "Domain mismatch between nodes and edges"
                raise ValueError(msg)

    @abstractmethod
    def connect(self) -> None:
        """Establish a connection to the knowledge graph database."""
        pass

    @abstractmethod
    def clean_database(self) -> None:
        """Delete all nodes, relationships, and indices from the graph."""
        pass

    @abstractmethod
    def create_indices(self) -> None:
        """Create indices for optimal query performance."""
        pass

    @abstractmethod
    def insert_nodes(self) -> int:
        """
        Insert nodes into the graph database.

        Returns:
            Number of nodes inserted
        """
        pass

    @abstractmethod
    def insert_edges(self) -> int:
        """
        Insert edges into the graph database.

        Returns:
            Number of edges inserted
        """
        pass

    @abstractmethod
    def verify_import(self) -> dict[str, int]:
        """
        Verify the import by counting nodes and edges.

        Returns:
            Dictionary with node and edge counts
        """
        pass

    def get_edges_tsv(self) -> pl.DataFrame:
        """Get the edges/facts TSV data."""
        return self.edges_df

    def ingest_from_s3(self, recreate: bool = False) -> None:
        """Ingest nodes and edges data from S3 paths."""
        self.download_data()

        self.ingest(
            nodes_file_path=str(self.nodes_tsv_path),
            edges_file_path=str(self.facts_tsv_path),
            recreate=recreate,
        )

    def ingest(
        self, nodes_file_path: str, edges_file_path: str, recreate: bool = False
    ) -> dict[str, int]:
        """
        Complete ingestion pipeline.

        Args:
            nodes_file_path: Path to nodes TSV file
            edges_file_path: Path to edges TSV file
            recreate: Whether to clean the database before ingestion

        Returns:
            Dictionary with ingestion statistics
        """
        # Clean if requested
        if recreate:
            self.clean_database()

        # Load data
        self.load_data(nodes_file_path, edges_file_path)

        # Validate
        self.validate_data()

        # Create indices
        self.create_indices()

        # Insert nodes
        nodes_inserted = self.insert_nodes()

        # Insert edges
        edges_inserted = self.insert_edges()

        # Verify
        stats = self.verify_import()

        # Save backup
        self.save()

        # Final summary
        logger.info("=" * 60)
        logger.info("✅ INGESTION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Graph: {self.graph_name}")
        logger.info(f"Nodes processed: {nodes_inserted:,}")
        logger.info(f"Edges processed: {edges_inserted:,}")
        logger.info(f"Domain types: {len(self.domain_types)}")
        logger.info(f"Relation types: {len(self.relation_types)}")
        logger.info("=" * 60)

        return {
            "nodes_processed": nodes_inserted,
            "edges_processed": edges_inserted,
            "nodes_in_db": stats.get("nodes", -1),
            "edges_in_db": stats.get("edges", -1),
            "domain_types": len(self.domain_types),
            "relation_types": len(self.relation_types),
        }

    def save(self) -> None:
        pass

    @abstractmethod
    def query(
        self, cypher_query: str, params: dict | None = None, timeout: int | None = None
    ) -> Any:
        """Execute a query on the graph."""
        pass

    def sql_query(self, query: str, tables: list[str] | None = None) -> list[dict]:
        """
        Execute a SQL query using Polars SQLContext on in-memory DataFrames.

        Supports querying `nodes`, `edges`, or both simultaneously via aliases.

        Args:
            query: SQL query string to execute
            tables: Optional names of tables. Defaults to ['nodes', 'edges']

        Returns:
            List of dictionaries representing query results

        Raises:
            DataNotLoadedError: If required DataFrames are not loaded
            ValueError: If query is invalid or tables are misconfigured
            pl.SQLSyntaxError: If SQL syntax is incorrect
        """

        if tables is None:
            tables = ["nodes", "edges"]
        table_dfs: dict[str, pl.DataFrame] = {}

        if "nodes" in tables:
            table_dfs["nodes"] = self.nodes_df
        if "edges" in tables:
            table_dfs["edges"] = self.edges_df

        if len(table_dfs) == 0:
            msg = f"Invalid 'tables' parameter: {tables}"
            raise ValueError(msg)

        sql_context = pl.SQLContext(
            edges=table_dfs.get("edges"), nodes=table_dfs.get("nodes"), eager=True
        )
        output_df = sql_context.execute(query)

        return output_df.to_dicts()

    @abstractmethod
    def find_all_shortest_paths(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 10,
        limit: int = 10,
        timeout: int | None = None,
    ) -> list:
        """Find all shortest paths between two nodes."""
        pass

    @abstractmethod
    def find_shortest_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 10,
        timeout: int | None = None,
    ) -> list[dict]:
        """Find the shortest path between two nodes."""
        pass

    def set_result_limit(self, limit: int = -1) -> None:
        """Set the maximum result set size for queries."""
        logger.info(
            f" {self.__class__.__name__} doesn't support global RESULTSET_SIZE configuration. "
            f"Setting limit to {limit}. "
            f"Use LIMIT clauses in queries instead."
        )

    def get_result_limit(self) -> int:
        """Get the current result set size limit."""
        logger.info(
            f"  {self.__class__.__name__} doesn't support global RESULTSET_SIZE configuration. "
            "Use LIMIT clauses in queries instead."
        )
        return -1

    def _batch_iterator(self, df: pl.DataFrame) -> Iterator[pl.DataFrame]:
        """Yield batches of dataframe for efficient processing."""
        for i in range(0, len(df), self.batch_size):
            yield df.slice(i, self.batch_size)

    @classmethod
    @abstractmethod
    def from_settings(cls, settings: T_Settings) -> "TemporalKGEngine[T_Settings]":
        pass
