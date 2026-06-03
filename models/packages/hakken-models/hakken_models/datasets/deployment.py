from __future__ import annotations

import fsspec
import numpy as np
import polars as pl
import torch
from loguru import logger

from hakken_models.core.constants import MissingPolicy
from hakken_models.core.entities.kg_data import KGData

RelationIndex = int
NodeIndex = int
DomainIndex = int

FactIndex = tuple[NodeIndex, RelationIndex, NodeIndex]
FactWithDomainsIndex = tuple[NodeIndex, RelationIndex, NodeIndex, DomainIndex, DomainIndex]


class DatasetDeployment:
    """
    Dataset-level deployment object.

    This class provides convenient access to a packaged knowledge graph dataset
    consisting of:

    * node / relation / timestamp / domain mapping tables
    * preprocessed triple tensors (train/val/test splits)
    * pre-computed node and relation embeddings (optional)
    * SQL-like querying of mapping tables using Polars

    A `DatasetDeployment` instance is bound to one dataset version
    `<DATASET_NAME>-<DATASET_VERSION>` stored under `target_root`.

    Directory layout::

        target_root/
            mappings/
                nodes_map.parquet
                relations_map.parquet
                timestamps_map.parquet
                domains_map.parquet
            tensors/
                train.npy
                val.npy
                test.npy
            embeddings/          # optional
                nodes.npy        # [num_nodes, embedding_dim]
                relations.npy    # [num_relations, embedding_dim]

    Precomputed multi-hot relation label tensors are loaded when present::

        tensors/
            train_relation_labels.npy  # [num_train, num_relations]
            val_relation_labels.npy    # [num_val,   num_relations]
    """

    def __init__(
        self,
        target_root: str = "s3://sai-spaice-ds/data/processed/data_processing/zenml/dataset_name/dataset_version",
        load_embeddings: bool = True,
    ) -> None:
        """
        Initialize a dataset deployment and load all artifacts.

        Loads mapping parquet files and fact tensors from the given root path.
        Local paths and remote filesystems (e.g. S3) are supported through
        `fsspec`.

        Args:
            target_root: Root directory containing `mappings/` and `tensors/`
                subfolders for this dataset version.
            load_embeddings: If False, skip loading pre-computed embeddings
                (nodes.npy, relations.npy). Use for KGE training where embeddings
                are not needed.

        Raises:
            FileNotFoundError: If any required mapping or tensor file is missing.
        """
        self.target_root = target_root
        logger.info(f"Loading dataset artifacts for {self.target_root}")

        # Define your storage configuration
        # TODO: This should be refactored to be more flexible and not hardcoded to S3 with specific credentials. Consider passing a storage configuration object instead.
        storage_options = {
            "region": "us-east-1",
            # If you aren't using an IAM role/profile, add your credentials here:
            # "aws_access_key_id": "<YOUR_KEY>",
            # "aws_secret_access_key": "<YOUR_SECRET>",
        }

        mappings_root = f"{self.target_root}/mappings"

        self.nodes_mapping_df: pl.DataFrame = pl.read_parquet(
            f"{mappings_root}/nodes_map.parquet", storage_options=storage_options
        )
        self.relations_mapping_df: pl.DataFrame = pl.read_parquet(
            f"{mappings_root}/relations_map.parquet", storage_options=storage_options
        )
        self.timestamps_mapping_df: pl.DataFrame = pl.read_parquet(
            f"{mappings_root}/timestamps_map.parquet", storage_options=storage_options
        )
        self.domains_mapping_df: pl.DataFrame = pl.read_parquet(
            f"{mappings_root}/domains_map.parquet", storage_options=storage_options
        )

        tensors_root = f"{self.target_root}/tensors"

        self._facts_np = {
            "train": self._load_npy(f"{tensors_root}/train.npy"),
            "val": self._load_npy(f"{tensors_root}/val.npy"),
            "test": self._load_npy(f"{tensors_root}/test.npy"),
        }

        logger.success("Mappings and tensors successfully loaded.")

        # ── optional pre-computed embeddings ─────────────────────────────
        if load_embeddings:
            embeddings_root = f"{self.target_root}/embeddings"
            self._node_embeddings_np: np.ndarray | None = self._load_npy_optional(
                f"{embeddings_root}/nodes.npy"
            )
            self._relation_embeddings_np: np.ndarray | None = self._load_npy_optional(
                f"{embeddings_root}/relations.npy"
            )
        else:
            self._node_embeddings_np: np.ndarray | None = None
            self._relation_embeddings_np: np.ndarray | None = None

        if self._node_embeddings_np is not None:
            expected_nodes = self.nodes_mapping_df.height
            actual_nodes = self._node_embeddings_np.shape[0]
            if actual_nodes != expected_nodes:
                raise ValueError(
                    f"Node embeddings row count ({actual_nodes}) does not match "
                    f"nodes mapping ({expected_nodes})"
                )
            logger.success(f"Node embeddings loaded: {self._node_embeddings_np.shape}")
        if self._relation_embeddings_np is not None:
            expected_rels = self.relations_mapping_df.height
            actual_rels = self._relation_embeddings_np.shape[0]
            if actual_rels != expected_rels:
                raise ValueError(
                    f"Relation embeddings row count ({actual_rels}) does not match "
                    f"relations mapping ({expected_rels})"
                )
            logger.success(f"Relation embeddings loaded: {self._relation_embeddings_np.shape}")

        # ── optional precomputed relation labels ─────────────────────────
        self._relation_labels_np: dict[str, np.ndarray] = {}
        for split in ("train", "val"):
            arr = self._load_npy_optional(f"{tensors_root}/{split}_relation_labels.npy")
            if arr is not None:
                self._relation_labels_np[split] = arr
                logger.success(f"Relation labels loaded ({split}): {arr.shape}")

        # ── timestamp index → year value lookup ──────────────────────────
        ts_df = self.timestamps_mapping_df.sort("index")
        self._t_idx_to_year: np.ndarray = ts_df["id"].to_numpy().astype(np.float32)

        # registry for SQL access
        self._tables: dict[str, pl.DataFrame] = {
            "nodes": self.nodes_mapping_df,
            "relations": self.relations_mapping_df,
            "timestamps": self.timestamps_mapping_df,
            "domains": self.domains_mapping_df,
        }

        self._node_id_to_index: dict[str, tuple[NodeIndex, DomainIndex]] = {
            row["id"]: (int(row["index"]), int(row["domain_index"]))
            for row in self.nodes_mapping_df.select(["id", "index", "domain_index"]).iter_rows(
                named=True
            )
        }

        self._relation_id_to_index: dict[str, RelationIndex] = {
            row["id"]: int(row["index"])
            for row in self.relations_mapping_df.select(["id", "index"]).iter_rows(named=True)
        }

        self._node_index_to_id: dict[NodeIndex, str] = {
            int(row["index"]): row["id"]
            for row in self.nodes_mapping_df.select(["index", "id"]).iter_rows(named=True)
        }
        self._relation_index_to_id: dict[RelationIndex, str] = {
            int(row["index"]): row["id"]
            for row in self.relations_mapping_df.select(["index", "id"]).iter_rows(named=True)
        }

        logger.success("Dataset loaded successfully")

    @property
    def num_entities(self) -> int:
        """Total number of distinct entities in the dataset."""
        return self.nodes_mapping_df.height

    @property
    def num_relations(self) -> int:
        """Total number of distinct relations in the dataset."""
        return self.relations_mapping_df.height

    @property
    def num_domains(self) -> int:
        """Total number of distinct domains associated with entities."""
        return self.domains_mapping_df.height

    @property
    def num_timestamps(self) -> int:
        """Total number of distinct timestamps in the dataset."""
        return self.timestamps_mapping_df.height

    @property
    def has_embeddings(self) -> bool:
        """Whether pre-computed node and relation embeddings are available."""
        return self._node_embeddings_np is not None and self._relation_embeddings_np is not None

    @property
    def embedding_dim(self) -> int | None:
        """Dimensionality of pre-computed embeddings, or ``None`` if unavailable."""
        if self._node_embeddings_np is None:
            return None
        return int(self._node_embeddings_np.shape[1])

    @property
    def has_relation_labels(self) -> bool:
        """Whether precomputed multi-hot relation labels are available."""
        return "train" in self._relation_labels_np and "val" in self._relation_labels_np

    def get_relation_labels_tensor(
        self, split_name: str, device: str | None = None
    ) -> torch.Tensor:
        """Return precomputed relation labels as a ``[N, R]`` float tensor.

        Raises:
            KeyError: If the split is not available or labels were not loaded.
        """
        if split_name not in self._relation_labels_np:
            available = list(self._relation_labels_np.keys()) or ["(none)"]
            raise KeyError(
                f"Relation labels not available for split '{split_name}'. Available: {available}"
            )
        return torch.tensor(
            self._relation_labels_np[split_name], dtype=torch.float32, device=device
        )

    # ── embedding & timestamp accessors ──────────────────────────────────

    def get_node_embedding_matrix(self, device: str | None = None) -> torch.Tensor:
        """Return node embeddings as a ``[num_nodes, embedding_dim]`` float tensor.

        Raises:
            RuntimeError: If node embeddings were not loaded.
        """
        if self._node_embeddings_np is None:
            raise RuntimeError(
                "Node embeddings not available. Place a nodes.npy file under "
                f"{self.target_root}/embeddings/"
            )
        return torch.tensor(self._node_embeddings_np, dtype=torch.float32, device=device)

    def get_relation_embedding_matrix(self, device: str | None = None) -> torch.Tensor:
        """Return relation embeddings as a ``[num_relations, embedding_dim]`` float tensor.

        Raises:
            RuntimeError: If relation embeddings were not loaded.
        """
        if self._relation_embeddings_np is None:
            raise RuntimeError(
                "Relation embeddings not available. Place a relations.npy file under "
                f"{self.target_root}/embeddings/"
            )
        return torch.tensor(self._relation_embeddings_np, dtype=torch.float32, device=device)

    def get_timestamp_values(self, device: str | None = None) -> torch.Tensor:
        """Return a ``[num_timestamps]`` float tensor mapping ``t_idx -> year``.

        The tensor is ordered by timestamp index so that
        ``result[t_idx]`` gives the original year value.
        """
        return torch.tensor(self._t_idx_to_year, dtype=torch.float32, device=device)

    def _load_npy(self, path: str) -> np.ndarray:
        """Load npy from local or S3 path."""
        logger.info(f"Loading tensor: {path}")
        with fsspec.open(path, "rb") as f:
            return np.load(f)

    def _load_npy_optional(self, path: str) -> np.ndarray | None:
        """Load npy if it exists, return ``None`` otherwise."""
        try:
            return self._load_npy(path)
        except FileNotFoundError:
            logger.debug(f"Optional file not found, skipping: {path}")
            return None

    def get_facts_tensor(self, split_name: str, device: str | None = None) -> torch.Tensor:
        """
        Get the facts tensor for a specific dataset split.

        Returns a PyTorch tensor containing the facts (triples) for the specified
        split. The tensor has shape [num_facts, 3] or [num_facts, 4] where each
        row represents a fact as (subject_index, relation_index, object_index)
        or (subject_index, relation_index, object_index, timestamp_index).

        Args:
            split_name: Name of the split to retrieve. Must be one of
                ``"train"``, ``"val"``, or ``"test"``.

        Returns:
            PyTorch tensor of shape [num_facts, 3] or [num_facts, 4] with dtype
            ``torch.long`` containing the indexed facts.

        Raises:
            KeyError: If the specified split name does not exist.
        """
        if split_name not in self._facts_np:
            raise KeyError(
                f"Unknown split name: {split_name}. Available splits: {list(self._facts_np.keys())}"
            )

        return torch.tensor(self._facts_np[split_name], device=device).long()

    def get_kg_data(self, split_names: list[str]) -> KGData:
        """
        Build a `KGData` object from one or more dataset splits.

        Concatenates the specified split tensors (train/val/test) and constructs
        a `KGData` instance with entity/domain metadata.

        Args:
            split_names: List of split names to include. Examples: `["train"]`,
                `["train", "val"]`, `["test"]`.

        Returns:
            A `KGData` object backed by PyTorch tensors.

        Raises:
            KeyError: If an unknown split name is provided.
        """
        facts_list = []
        for split_name in split_names:
            facts_i = torch.tensor(self._facts_np[split_name])
            if facts_i.numel():
                facts_list.append(facts_i)
            else:
                logger.warning(f"{split_name} facts does not have any element")

        facts_pt = torch.cat(facts_list, dim=0).long()

        return KGData.from_facts(
            facts=facts_pt,
            num_nodes=self.num_entities,
            domains_mapping_df=self.nodes_mapping_df.select(
                pl.col("index").alias("node_id"),
                pl.col("domain_index").alias("domain_id"),
            ),
            num_domains=self.num_domains,
            relabel_nodes=False,
        )

    def sample_random_facts(
        self,
        splits: list[str] = ("train",),
        num_samples: int = 10,
        replace: bool = True,
        rng: np.random.Generator | None = None,
    ) -> list[tuple[str, str, str]]:
        """
        Sample random facts (triples) from the specified dataset splits.

        Args:
            splits: List of split names (e.g. ``["train", "val"]``). Invalid splits
                are skipped.
            num_samples: Number of facts to sample.
            replace: Whether to sample with replacement.
            rng: Optional random number generator for reproducibility.

        Returns:
            List of ``(subject_id, relation_id, object_id)`` tuples.

        Raises:
            ValueError: If no facts are available in the requested splits, or if
                ``num_samples`` exceeds population when ``replace=False``.
        """
        np_arrays = []
        for split in splits:
            if split not in self._facts_np:
                continue
            arr = self._facts_np[split]
            if arr.size > 0:
                np_arrays.append(arr)

        if not np_arrays:
            raise ValueError(
                f"No facts available in requested splits {splits}. "
                f"Available splits: {list(self._facts_np.keys())}"
            )

        all_facts = np.concatenate(np_arrays, axis=0)
        total = all_facts.shape[0]

        if not replace and num_samples > total:
            raise ValueError(
                f"Cannot sample {num_samples} facts without replacement from "
                f"{total} available facts."
            )

        rng = rng if rng is not None else np.random.default_rng()
        idx = rng.choice(total, size=num_samples, replace=replace)
        sampled = all_facts[idx]

        result: list[tuple[str, str, str]] = []
        for row in sampled:
            s, r, o = int(row[0]), int(row[1]), int(row[2])
            result.append(
                (
                    self._node_index_to_id.get(s, "<UNK>"),
                    self._relation_index_to_id.get(r, "<UNK>"),
                    self._node_index_to_id.get(o, "<UNK>"),
                )
            )
        return result

    def sql_query(
        self,
        query: str,
        tables: list[str] | None = None,
    ) -> list[dict]:
        """
        Execute a SQL query over dataset mapping tables.

        Tables available by default:
        * ``nodes``
        * ``relations``
        * ``timestamps``
        * ``domains``

        Args:
            query: SQL query string (DuckDB/Polars SQL dialect).
            tables: Optional subset of table names to register for the query.
                If ``None``, all tables are available.

        Returns:
            List of result rows as dictionaries.

        Raises:
            ValueError: If any requested table does not exist.
        """
        if tables is None:
            tables = list(self._tables.keys())

        missing = set(tables) - set(self._tables.keys())
        if missing:
            raise ValueError(f"Unknown tables requested: {missing}")

        logger.debug(f"Executing SQL query on tables: {tables}")

        ctx = pl.SQLContext()
        for name in tables:
            ctx.register(name, self._tables[name])

        result = ctx.execute(query).collect()
        return result.to_dicts()

    def map_node_id_to_index(
        self, node_id: str, on_missing: MissingPolicy = MissingPolicy.ZERO
    ) -> tuple[NodeIndex, DomainIndex]:
        """
        Map a node string identifier to its numeric index and domain index.
        """
        try:
            return self._node_id_to_index[node_id]
        except KeyError as e:
            if on_missing is MissingPolicy.ZERO:
                return (-1, -1)
            if on_missing is MissingPolicy.RAISE:
                raise
            raise ValueError(f"Unknown on_missing policy: {on_missing}") from e

    def map_relation_id_to_index(
        self, relation_id: str, on_missing: MissingPolicy = MissingPolicy.ZERO
    ) -> RelationIndex:
        """
        Map a relation string identifier to its numeric index.
        """
        try:
            return self._relation_id_to_index[relation_id]
        except KeyError as e:
            if on_missing is MissingPolicy.ZERO:
                return -1
            if on_missing is MissingPolicy.RAISE:
                raise
            raise ValueError(f"Unknown on_missing policy: {on_missing}") from e

    def map_fact_ids_to_indexes(
        self,
        facts_list: list[tuple[str, str, str]],
        include_domains: bool = False,
        on_missing: MissingPolicy = MissingPolicy.ZERO,
    ) -> list[tuple]:
        """
        Map triples of string IDs to index-based facts.

        Args:
            facts_list: List of triples ``(subject_id, relation_id, object_id)``.
            include_domains: If ``True``, domain indices for subject and object
                are appended to each mapped triple:
                ``[sub, rel, obj, sub_domain, obj_domain]``.
            on_missing: Missing-ID handling policy applied to nodes and relations.

        Returns:
            List of mapped indexed triples (optionally with domain indices).
        """
        facts_index_list: list[tuple] = []
        for sub, rel, obj in facts_list:
            sub_idx, sub_domain_idx = self.map_node_id_to_index(sub, on_missing=on_missing)
            rel_idx = self.map_relation_id_to_index(rel, on_missing=on_missing)
            obj_idx, obj_domain_idx = self.map_node_id_to_index(obj, on_missing=on_missing)

            if include_domains:
                facts_index_list.append((sub_idx, rel_idx, obj_idx, sub_domain_idx, obj_domain_idx))
            else:
                facts_index_list.append((sub_idx, rel_idx, obj_idx))

        return facts_index_list

    def map_node_ids_to_indexes(
        self,
        node_id_list: list[str],
        on_missing: MissingPolicy = MissingPolicy.ZERO,
    ) -> list[NodeIndex]:
        """
        Map multiple node string IDs to their corresponding indices.
        """

        result = []

        for node_id in node_id_list:
            node_index = self.map_node_id_to_index(node_id, on_missing=on_missing)
            result.append(node_index[0])
        return result

    def map_node_ids_to_indexes_with_domains(
        self,
        node_id_list: list[str],
        on_missing: MissingPolicy = MissingPolicy.ZERO,
    ) -> list[tuple[NodeIndex, DomainIndex]]:
        """
        Map multiple node string IDs to their corresponding indices.
        """

        result = []

        for node_id in node_id_list:
            node_index = self.map_node_id_to_index(node_id, on_missing=on_missing)
            result.append(node_index)
        return result

    def get_relation_ids(self) -> list[str]:
        """
        Return all relation identifiers known to the dataset.

        Returns:
            List of external relation ID strings.
        """
        return list(self._relation_id_to_index.keys())

    def map_relation_ids_to_indexes(
        self, relation_id_list: list[str], on_missing: MissingPolicy = MissingPolicy.ZERO
    ) -> list[RelationIndex]:
        """
        Map multiple relation string IDs to numeric indices.
        """
        result: list[RelationIndex] = []

        for relation_id in relation_id_list:
            rel_idx = self.map_relation_id_to_index(relation_id, on_missing=on_missing)
            result.append(rel_idx)

        return result
