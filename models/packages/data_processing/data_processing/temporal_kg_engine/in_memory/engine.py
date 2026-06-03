import pickle
from pathlib import Path

import boto3
import networkx as nx
from botocore.exceptions import ClientError, NoCredentialsError
from loguru import logger

from data_processing.temporal_kg_engine.base import TemporalKGEngine
from data_processing.temporal_kg_engine.exceptions import ConfigurationError
from data_processing.temporal_kg_engine.in_memory.fact_path_finder import FactPathFinder
from data_processing.utils.timeout import timeout_fn

from .settings import InMemoryTKGSettings

NodeID = str
RelationID = str


class InMemoryTKGEngine(TemporalKGEngine[InMemoryTKGSettings]):
    """
    In-memory implementation of the Temporal Knowledge Graph Engine.

    Uses an in-memory MultiDiGraph for graph operations. This implementation
    is suitable for small to medium-sized graphs that fit in memory.

    Edge keys are set to relation_type, with year and occurrences stored as
    edge attributes.
    """

    def __init__(self, *args, **kwargs):
        """
        In-memory implementation of the Temporal Knowledge Graph Engine.

        Uses an in-memory MultiDiGraph for graph operations. This implementation
        is suitable for small to medium-sized graphs that fit in memory.

        Edge keys are set to relation_type, with year and occurrences stored as
        edge attributes.

        Supports loading graph from local cache or S3.
        """
        super().__init__(*args, **kwargs)

        # NetworkX directed multigraph (in-memory)
        self._graph: nx.MultiDiGraph | None = None

        logger.info(f"Initialized InMemoryTKGEngine for graph: {self.graph_name}")

    @property
    def graph(self) -> nx.MultiDiGraph:
        """Get the NetworkX MultiDiGraph instance."""
        if self._graph is None:
            self._graph = nx.MultiDiGraph()
        return self._graph

    @property
    def graph_backup_path(self) -> Path:
        if self.local_data_storage_dir is None:
            raise ConfigurationError.missing(
                attr="local_data_storage_dir",
                context={"graph_name": self.graph_name},
                hint=(
                    "Provide a path template containing '{graph_name}', e.g. "
                    "'/var/lib/temporal_kg/graphs/{graph_name}'."
                ),
            )
        folder = Path(self.local_data_storage_dir.format(graph_name=self.graph_name))

        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{self.graph_name}.pkl"

    @property
    def s3_graph_backup_path(self) -> str:
        if self.s3_backup_dir is not None:
            s3_backup_dir = self.s3_backup_dir.format(graph_name=self.graph_name)
            return f"{s3_backup_dir}/{self.graph_name}.pkl"
        raise ConfigurationError.missing(
            attr="s3_backup_dir",
            context={"graph_name": self.graph_name},
            hint="Provide a path template containing '{graph_name}', e.g. 's3://my-bucket/graphs/{graph_name}'.",
        )

    def download_backup(self) -> None:
        if self.s3_backup_dir is None:
            logger.info("S3 backup directory not configured. Skipping S3 download.")
            return
        bucket_name, s3_key = self._parse_s3_path(self.s3_graph_backup_path)

        try:
            logger.info(f"Attempting to download graph from S3: {self.s3_graph_backup_path}")

            s3_client = boto3.client("s3")
            s3_client.download_file(bucket_name, s3_key, str(self.graph_backup_path))

        except NoCredentialsError:
            logger.warning("AWS credentials not found. Unable to download graph from S3.")
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            if error_code in {"404", "NoSuchKey"}:
                logger.warning(f"Graph file not found in S3: {self.s3_graph_backup_path}")
            else:
                logger.warning(f"Failed to download graph from S3: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error downloading from S3: {e}")
        else:
            logger.info(f"✅ Downloaded graph from S3: s3://{bucket_name}/{s3_key}")

    def load_graph(self) -> bool:
        graph_loaded = False

        # Step 1: Try to load from local cache
        if self.graph_backup_path:
            try:
                if self.graph_backup_path.exists():
                    logger.info(f"Found local cache file: {self.graph_backup_path}")
                    with open(self.graph_backup_path, "rb") as f:
                        self._graph = pickle.load(f)
                    logger.info(f"✅ Loaded graph from local cache: {self.graph_backup_path}")
                    logger.info(f"   Nodes: {self.graph.number_of_nodes():,}")
                    logger.info(f"   Edges: {self.graph.number_of_edges():,}")
                    graph_loaded = True
            except Exception as e:
                logger.warning(f"Failed to load graph from local cache: {e}")

        return graph_loaded

    def connect(self) -> None:
        """
        Initialize the in-memory graph.

        Attempts to load from local cache first, then tries to download from S3
        if local cache doesn't exist. If both fail, logs a warning and continues
        with an empty graph.
        """
        logger.info("=" * 60)
        logger.info("INITIALIZING IN-MEMORY GRAPH")
        logger.info("=" * 60)

        self.download_data()
        if not self.graph_backup_path.exists():
            self.download_backup()

        # Initialize empty graph if nothing else works
        graph_loaded = self.load_graph()

        # Step 3: If everything failed, initialize empty graph
        if not graph_loaded:
            logger.warning(
                "Could not load graph from local cache or S3. "
                "Proceeding with empty graph. "
                "Use ingest() to populate the graph."
            )
            if self._graph is None:
                self._graph = nx.MultiDiGraph()

        logger.info("=" * 60)

    def clean_database(self) -> None:
        """Delete all nodes and edges from the graph."""
        logger.info("=" * 60)
        logger.info("CLEANING GRAPH")
        logger.info("=" * 60)

        if self._graph is not None:
            node_count = self.graph.number_of_nodes()
            edge_count = self.graph.number_of_edges()

            self.graph.clear()
            self._graph = None

            logger.info(f"✅ Deleted {node_count:,} nodes and {edge_count:,} edges")
            logger.info(f"✅ Graph {self.graph_name} is completely clean")
        else:
            logger.info("✅ Graph is already empty")

        logger.info("=" * 60)

    def create_indices(self) -> None:
        """
        NetworkX doesn't require explicit index creation.
        Indices are maintained automatically).
        """
        logger.info("=" * 60)
        logger.info("STEP 3: Creating Indices")
        logger.info("=" * 60)

        logger.info("✅ NetworkX maintains indices automatically - no explicit creation needed")
        logger.info("=" * 60)

    def insert_nodes(self) -> int:
        """
        Insert nodes into the graph database.

        Returns:
            Number of nodes inserted
        """
        logger.info("=" * 60)
        logger.info("STEP 4: Inserting Nodes")
        logger.info("=" * 60)

        total_nodes_inserted = 0

        for batch_idx, batch in enumerate(self._batch_iterator(self.nodes_df)):
            # Group by domain for efficient insertion

            for row in batch.iter_rows(named=True):
                node_data = {}

                # Map dataframe columns to graph properties
                for df_col, graph_prop in self.node_properties.items():
                    if df_col in row:
                        node_data[graph_prop] = row[df_col]

                node_data["domain"] = row["node_domain"]
                self.graph.add_node(row["node_id"], **node_data)
                total_nodes_inserted += 1

            # Progress logging
            if (batch_idx + 1) % 10 == 0:
                progress = (total_nodes_inserted / len(self.nodes_df)) * 100
                logger.info(
                    f"  Progress: {total_nodes_inserted:,}/{len(self.nodes_df):,} "
                    f"nodes ({progress:.1f}%)"
                )

        logger.info(f"✅ Inserted {total_nodes_inserted:,} nodes")
        return total_nodes_inserted

    def insert_edges(self) -> int:
        """
        Insert edges into the graph database.

        Uses relation_type as the edge key, with year and occurrences as attributes.
        If multiple edges with the same relation_type exist between the same nodes,
        NetworkX will auto-generate unique keys (0, 1, 2, ...) while preserving
        the relation_type in the edge data.

        Returns:
            Number of edges inserted
        """
        logger.info("=" * 60)
        logger.info("STEP 5: Inserting Edges")
        logger.info("=" * 60)

        total_edges_inserted = 0

        for batch_idx, batch in enumerate(self._batch_iterator(self.edges_df)):
            for row in batch.iter_rows(named=True):
                subject_id = row["subject_id"]
                object_id = row["object_id"]
                relation_type = row["relation_type"]

                edge_attrs = {
                    "relation_type": relation_type,
                    "subject_domain": row["subject_domain"],
                    "object_domain": row["object_domain"],
                    "year": int(row["year"]),
                    "occurrences": int(row["number_of_occurrences"]),
                }

                self.graph.add_edge(subject_id, object_id, key=relation_type, **edge_attrs)
                total_edges_inserted += 1
            # Progress logging
            if (batch_idx + 1) % 10 == 0:
                progress = (total_edges_inserted / len(self.edges_df)) * 100
                logger.info(
                    f"  Progress: {total_edges_inserted:,}/{len(self.edges_df):,} "
                    f"edges ({progress:.1f}%)"
                )

        logger.info(f"✅ Inserted {total_edges_inserted:,} edges")
        return total_edges_inserted

    def verify_import(self) -> dict[str, int]:  # noqa: PLR0912
        """
        Verify the import by counting nodes and edges.

        Returns:
            Dictionary with node and edge counts
        """
        logger.info("=" * 60)
        logger.info("STEP 6: Verifying Import")
        logger.info("=" * 60)

        stats = {}

        # Count nodes
        try:
            count = self.graph.number_of_nodes()
            stats["nodes"] = count
            logger.info(f"Total nodes in graph: {stats['nodes']:,}")
        except Exception as e:
            logger.error(f"Failed to count nodes: {e}")
            stats["nodes"] = -1

        # Count edges
        try:
            count = self.graph.number_of_edges()
            stats["edges"] = count
            logger.info(f"Total edges in graph: {stats['edges']:,}")
        except Exception as e:
            logger.error(f"Failed to count edges: {e}")
            stats["edges"] = -1

        # Verify nodes by domain
        logger.info("\nNodes by domain:")
        for domain in self.domain_types:
            try:
                count = sum(
                    1 for n in self.graph.nodes() if self.graph.nodes[n].get("domain") == domain
                )
                logger.info(f"  {domain}: {count:,}")
            except Exception as e:
                logger.warning(f"  {domain}: count failed - {e}")

        # Get relationship types in the graph
        logger.info("\nRelationship types in graph:")
        try:
            rel_types = set()
            for _, _, edge_data in self.graph.edges(data=True):
                rel_type = edge_data.get("relation_type")
                if rel_type:
                    rel_types.add(rel_type)

            for rel_type in sorted(rel_types)[:20]:
                logger.info(f"  {rel_type}")
        except Exception as e:
            logger.debug(f"Could not list relationship types: {e}")
            logger.info("Relation types from data:")
            for rel_type in self.relation_types[:10]:
                logger.info(f"  {rel_type}")

        # Sample edges by relationship type
        logger.info("\nSample edges by relationship type:")
        try:
            sample_relation_types = self.relation_types[:3]
            for rel_type in sample_relation_types:
                sample_edges = [
                    (s, o, key, data)
                    for s, o, key, data in self.graph.edges(keys=True, data=True)
                    if data.get("relation_type") == rel_type
                ][:2]

                if sample_edges:
                    logger.info(f"\n  Relationship: {rel_type}")
                    for s, o, key, data in sample_edges:
                        logger.info(
                            f"    {s} --[{data.get('relation_type')}]--> "
                            f"{o} (year: {data.get('year')}, "
                            f"occ: {data.get('occurrences')}, key: {key})"
                        )
                else:
                    logger.warning(f"  No edges found for {rel_type}")
        except Exception as e:
            logger.warning(f"Sample query failed: {e}")

        return stats

    def write_graph(self) -> None:
        with open(self.graph_backup_path, "wb") as f:
            pickle.dump(self._graph, f)

    def save(self) -> None:
        """
        Save the current graph to a local file and upload it to S3.

        Saves the graph using NetworkX pickle format to maintain compatibility
        with the load_graph method. If S3 backup is configured, uploads the
        saved file to S3 after successful local save.
        """
        if self._graph is None:
            logger.warning("No graph to save.")
            return

        # Step 1: Save to local file
        try:
            logger.info(f"Attempting to save graph locally: {self.graph_backup_path}")
            self.write_graph()
            logger.info(f"✅ Saved graph to local cache: {self.graph_backup_path}")
            logger.info(f"   Nodes: {self.graph.number_of_nodes():,}")
            logger.info(f"   Edges: {self.graph.number_of_edges():,}")
        except Exception as e:
            logger.error(f"Failed to save graph to local cache: {e}")
            return

        # Step 2: Upload to S3 if configured
        if self.s3_backup_dir and self.s3_graph_backup_path:
            try:
                logger.info(f"Attempting to upload graph to S3: {self.s3_graph_backup_path}")
                bucket_name, s3_key = self._parse_s3_path(self.s3_graph_backup_path)
                s3_client = boto3.client("s3")
                s3_client.upload_file(str(self.graph_backup_path), bucket_name, s3_key)
                logger.info(f"✅ Uploaded graph to S3: s3://{bucket_name}/{s3_key}")
            except NoCredentialsError:
                logger.warning("AWS credentials not found. Unable to upload graph to S3.")
            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "Unknown")
                logger.warning(f"Failed to upload graph to S3: {e} (Error code: {error_code})")
            except Exception as e:
                logger.warning(f"Unexpected error uploading to S3: {e}")
        else:
            logger.info("S3 backup directory not configured. Skipping S3 upload.")

    def query(
        self, cypher_query: str, params: dict | None = None, timeout: int | None = None
    ) -> list:
        """
        NetworkX doesn't support Cypher queries.

        This method raises NotImplementedError. For querying NetworkX graphs,
        use the graph API directly or implement custom query methods.

        Args:
            cypher_query: Cypher query string (not supported)
            params: Query parameters (not supported)
            timeout: Query timeout (not applicable)

        Returns:
            Never returns (raises NotImplementedError)

        Raises:
            NotImplementedError: NetworkX doesn't support Cypher queries
        """
        msg = (
            f"NetworkX doesn't support Cypher queries. "
            f"Use the graph API directly or implement custom query methods. "
            f"Query: {cypher_query[:100]}..."
        )
        raise NotImplementedError(msg)

    def find_all_shortest_paths(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 0,
        limit: int = 10,
        timeout: int | None = None,
    ) -> list[list[NodeID]]:
        """
        Find all shortest paths between two nodes.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            max_depth: Maximum path length to search
            limit: Maximum number of paths to return
            timeout: Query timeout

        Returns:
            List of path dictionaries with node_ids, node_types, rel_types, etc.
        """
        if source_id not in self.graph:
            logger.warning(f"Source node {source_id} not found in graph")
            return []

        if target_id not in self.graph:
            logger.warning(f"Target node {target_id} not found in graph")
            return []

        if max_depth > 0:
            logger.warning("max_depth parameter is not enforced in NetworkX implementation")

        try:
            paths = []

            def _find_shortest_paths():
                for i, path in enumerate(
                    nx.all_shortest_paths(self.graph, source=source_id, target=target_id)
                ):
                    paths.append(path)
                    if i + 1 >= limit:
                        break

            if timeout is not None and timeout > 0:
                _find_shortest_paths = timeout_fn(seconds=timeout)(_find_shortest_paths)
            _find_shortest_paths()

        except nx.NetworkXNoPath:
            logger.info(f"No path found between {source_id} and {target_id}")
            return []
        except Exception as e:
            logger.error(f"Error finding shortest paths: {e}")
            return []
        else:
            return paths

    def find_all_shortest_fact_paths(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 0,
        limit: int = 10,
        _timeout: int | None = None,
    ) -> list[list[tuple[NodeID, RelationID, NodeID]]]:
        node_paths = self.find_all_shortest_paths(
            source_id=source_id, target_id=target_id, max_depth=max_depth, limit=limit, timeout=None
        )
        fact_path_finder = FactPathFinder(graph=self.graph)

        fact_paths = []
        for node_path in node_paths:
            fact_paths_i = fact_path_finder.node_path_to_facts(node_path=node_path, max_paths=limit)
            fact_paths.extend(fact_paths_i)
            if len(fact_paths) >= limit:
                break

        return fact_paths[:limit]

    def find_shortest_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 10,
        timeout: int | None = None,
    ) -> list[dict]:
        raise NotImplementedError()

    def close(self) -> None:
        """Clear the graph (NetworkX doesn't require persistent connection cleanup)."""
        if self._graph is not None:
            self._graph.clear()
            self._graph = None
            logger.info("✅ Cleared NetworkX graph")

    def __enter__(self):
        """Context manager entry."""
        if self._graph is None:
            self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __repr__(self) -> str:
        node_count = self.graph.number_of_nodes() if self._graph is not None else 0
        edge_count = self.graph.number_of_edges() if self._graph is not None else 0
        return (
            f"InMemoryTKGEngine(graph_name='{self.graph_name}', "
            f"batch_size={self.batch_size}, "
            f"nodes={node_count:,}, edges={edge_count:,})"
        )

    @classmethod
    def from_settings(cls, settings: InMemoryTKGSettings) -> "InMemoryTKGEngine":
        """Create instance from settings."""
        return cls(**cls._base_args(settings))
