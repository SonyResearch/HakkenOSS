from __future__ import annotations

import hashlib
import json
import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar, cast

import pandas as pd
from hakken_ml_toolkit.file_manager import S3Manager
from hakken_ml_toolkit.ml_base_structures import KnowledgeGraph
from loguru import logger
from omegaconf import MISSING
from pydantic import BaseModel, ConfigDict, Field, field_validator

from datasets.common.exceptions import (
    DataSplitProportionError,
    GraphNotLoadedError,
    KnowledgeGraphObjectError,
)


class DataRepositoryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    root_folder: str = Field(
        default_factory=lambda: os.environ.get("DATA_PATH", "./data"),
        description="Local root folder containing the data.",
    )
    cache_folder: str | None = Field(
        default=None,
        description="Local folder for on-disk kg data. Created if provided.",
    )
    s3_cache_folder: str | None = Field(
        default=None,
        description="Optional S3 folder to mirror cache from/to.",
    )
    data_split_proportion_dict: dict[str, float] | None = Field(
        default=None,
        description="Mapping split name -> proportion in [0,1]. Must sum to 1.0.",
        examples=[{"train": 0.8, "valid": 0.1, "test": 0.1}],
    )
    data_split_temporal_dict: dict[str, tuple[int, int]] | None = Field(
        default=None,
        description="Mapping split name -> (start, end) timestamp inclusive.",
        examples=[{"train": (2000, 2015), "valid": (2016, 2018), "test": (2019, 2020)}],
    )

    relation_filter: list[str] | None = Field(
        default=None,
        description="Optional allow-list of relation names.",
    )

    @field_validator("data_split_proportion_dict")
    @classmethod
    def validate_split_proportions(cls, v: dict[str, float] | None) -> dict[str, float] | None:
        """Validate that data split proportions sum to 1.0"""
        if v is not None and abs(sum(v.values()) - 1.0) > 1e-10:
            msg = f"Data split proportions must sum to 1.0, got {sum(v.values())}"
            raise DataSplitProportionError(msg)
        return v

    @field_validator("data_split_temporal_dict")
    @classmethod
    def validate_temporal_ranges(
        cls, v: dict[str, tuple[int, int]] | None
    ) -> dict[str, tuple[int, int]] | None:
        """Validate that each temporal range is a (start, end) tuple with start <= end."""
        if v is None:
            return v

        for split, bounds in v.items():
            if not isinstance(bounds, tuple) or len(bounds) != 2:
                msg = (
                    f"data_split_temporal_dict['{split}'] must be a tuple[int, int]"
                    "that represents (start, end)."
                    "Got: {bounds!r}"
                )
                raise ValueError(msg)
            start, end = bounds
            if not isinstance(start, int) or not isinstance(end, int):
                msg = (
                    f"data_split_temporal_dict['{split}'] must contain integers."
                    "Got: ({start!r}, {end!r})"
                )
                raise TypeError(msg)
            if start > end:
                msg = (
                    f"data_split_temporal_dict['{split}'] has start ({start}) > end ({end}). "
                    f"Start must be <= end."
                )
                raise ValueError(msg)
        return v

    def get_excluded_fields(self) -> set[str]:
        return {"cache_folder", "root_folder", "s3_cache_folder"}

    def __hash__(self) -> int:
        return hash(self.md5_hash())

    def md5_hash(self) -> str:
        excluded_fields = self.get_excluded_fields()

        hash_values = []
        fields = sorted(
            field for field in self.__class__.model_fields if field not in excluded_fields
        )
        for field in fields:
            value = getattr(self, field, None)
            # Handle MISSING sentinel value
            if value is not MISSING and value is not None:
                hash_values.append(str(value))

        config_str = "--".join(hash_values)

        return hashlib.md5(config_str.encode()).hexdigest()


T = TypeVar("T", bound=DataRepositoryConfig)


class DataRepositoryI(ABC, Generic[T]):
    """
    Interface for data repositories that manage knowledge graphs.

    This abstract base class provides a standardized way to load, cache, and access
    knowledge graph data. It handles configuration validation, caching mechanisms,
    and common access patterns for working with graph-structured data.

    The class uses generic typing with type parameter T, which must be a subclass
    of DataRepositoryConfig. This allows for type-safe specialization where concrete
    implementations can specify their exact configuration type requirements. When
    subclassing, you can define a specific configuration class that extends the base
    DataRepositoryConfig to add domain-specific settings.

    Attributes:
        name: A string identifier for the repository.
        config: Configuration settings for the repository.
        kg: The loaded knowledge graph instance, if any.
        cache_dir: Directory for caching the loaded graph.
        use_cache: Whether to use cached data when available.
    """

    name: str

    def __init__(self, config: T):
        self.config = config
        self.kg: KnowledgeGraph | None = None

        self._cache_dir: Path | None = None
        if config.cache_folder:
            self._cache_dir = Path(config.cache_folder)
            self._cache_dir.mkdir(parents=True, exist_ok=True)

        self.s3_cache_dir = None
        if self.config.s3_cache_folder is not None:
            self.s3_cache_dir = Path(self.config.s3_cache_folder)

    @property
    def use_cache(self) -> bool:
        return self._cache_dir is not None

    @property
    def config_hash_file(self) -> Path:
        if self.use_cache:
            return cast("Path", self.cache_dir / "config_hash.txt")

        msg = "cache_dir is not configured; cannot compute config_hash_file"
        raise ValueError(msg)

    @property
    def cache_dir(self) -> Path:
        if self._cache_dir is not None:
            return self._cache_dir

        msg = "cache_dir is not configured. Set DataRepositoryConfig.cache_folder"
        raise AttributeError(msg)

    @property
    def num_relations(self) -> int:
        if self.kg is None:
            raise GraphNotLoadedError()
        if self.kg.num_relations is not None:
            return int(self.kg.num_relations)
        raise ValueError()

    @property
    def num_entities(self) -> int:
        if self.kg is None:
            raise GraphNotLoadedError()

        if self.kg.num_entities is not None:
            return int(self.kg.num_entities)

        raise ValueError()

    @property
    def num_timestamps(self) -> int | None:
        if self.kg is None:
            raise GraphNotLoadedError()
        num_ts = self.kg.num_timestamps
        return int(num_ts) if num_ts is not None else None

    @property
    def num_domains(self) -> int | None:
        if self.kg is None:
            raise GraphNotLoadedError()
        return self.kg.num_domains

    @classmethod
    @abstractmethod
    def _get_config_class(cls) -> type[T]:
        """Extract the config type from the class's Generic parameters"""
        pass

    def prune_invalid_facts(
        self,
        facts_df: pd.DataFrame,
        time_col: str | None = "year",
        subject_id_col: str = "subject_id",
        object_id_col: str = "object_id",
    ) -> pd.DataFrame:
        if self.config.data_split_proportion_dict is not None:
            train_proportion = self.config.data_split_proportion_dict["train"]
            end_idx = int(len(facts_df) * train_proportion)
            train_facts_df = facts_df.iloc[:end_idx]
        elif self.config.data_split_temporal_dict is not None:
            if time_col is None:
                raise RuntimeError()
            min_year, max_year = self.config.data_split_temporal_dict["train"]

            cond = facts_df[time_col].between(min_year, max_year)
            train_facts_df = facts_df.loc[cond]
        else:
            raise NotImplementedError()

        train_entities = pd.concat(
            [train_facts_df[subject_id_col], train_facts_df[object_id_col]]
        ).unique()

        entity_cols = [subject_id_col, object_id_col]

        cond = facts_df[entity_cols].isin(train_entities).all(axis=1)

        return facts_df.loc[cond]

    def load_data(self) -> KnowledgeGraph:
        if self.kg is None:
            # Try to load from cache first
            cached_kg = self._load_from_cache()
            if cached_kg is not None:
                logger.info(f"Loaded knowledge graph from cache {self.cache_dir}")
                self.kg = cached_kg
            else:
                logger.info("Cache not found or expired. Loading from database...")
                kg = self._load_from_database()

                if isinstance(kg, KnowledgeGraph):
                    self.kg = kg
                else:
                    raise KnowledgeGraphObjectError()

                if self.use_cache:
                    # Save to cache for future use
                    self._save_to_cache(kg)

        return self.kg

    def clear_cache(self) -> None:
        """
        Clear the cached knowledge graph data for the current configuration.

        This method removes the cache directory for the current repository configuration,
        if it exists. Subsequent calls to load_data() will load from the primary data source
        rather than the cache until a new cache is created.

        Note that this only affects the cache for the current configuration hash.
        Other cached configurations for the same primary data source remain untouched.
        """
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            logger.info(f"Cache cleared for {self.cache_dir}")

    def _load_from_cache(self) -> KnowledgeGraph | None:
        """
        Attempt to load the knowledge graph from cache.

        This method checks if a cached version exists and loads it if available.
        Note that a primary data source can have multiple cache versions,
        each identified by the hash of its configuration parameters.

        Returns:
            The cached knowledge graph if successful, None otherwise.
        """
        if not self.use_cache:
            return None

        if self.s3_cache_dir is not None:
            s3_manager = S3Manager()

            cache_file = s3_manager.find(str(self.s3_cache_dir))
            if len(cache_file) == 0:
                return None

            s3_manager.download_folder(
                remote_path=str(self.s3_cache_dir), local_path=str(self.cache_dir)
            )

        try:
            kg = KnowledgeGraph.load(self.cache_dir)
        except Exception as e:
            logger.info(f"Error loading from cache: {e!s}")
            return None
        else:
            stored_hash = self.config_hash_file.read_text()
            current_hash = self.config.md5_hash()
            if stored_hash == current_hash:
                return kg
            logger.info(
                f"Cache invalid: stored hash {stored_hash} != current hash {current_hash}. "
                "Will ignore cached data and load from the primary source."
            )
            return None

    @abstractmethod
    def _load_from_database(self) -> KnowledgeGraph:
        """
        Load the knowledge graph from its primary data source.

        Returns:
            A loaded knowledge graph instance.
        """
        pass

    def _save_to_cache(self, kg: KnowledgeGraph) -> None:
        kg.save(self.cache_dir)

        self.config_hash_file.write_text(self.config.md5_hash())

        logger.info("Saved knowledge graph to cache.")

    def save_config(self, json_path: str | Path):
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)

        with open(str(json_path), "w") as f:
            json.dump(self.config.model_dump_json(), f)

    @classmethod
    def load_config(cls, json_path: str | Path) -> T:
        path = Path(json_path)
        if not path.exists():
            msg = f"{json_path} file does not exists"
            raise FileExistsError(msg)
        with open(str(json_path)) as f:
            data = json.load(f)

        config_class = cls._get_config_class()

        return config_class.model_validate_json(data)

    @classmethod
    def load(cls, folder_path: str | Path, config_update: dict | None = None) -> DataRepositoryI:
        config_path = Path(folder_path) / "config.json"

        config = cls.load_config(config_path)
        if config_update is not None:
            config = config.model_copy(update=config_update)
        return cls(config)

    def save(self, folder_path: str | Path):
        config_path = Path(folder_path) / "config.json"
        self.save_config(config_path)
