from __future__ import annotations

# ruff: noqa: TC003
from pathlib import Path
from typing import Any

from omegaconf import MISSING

from datasets.data_repo.base import DataRepositoryConfig


class DSVKGConfig(DataRepositoryConfig):
    facts_file: str | Path = MISSING
    facts_file_delimiter: str = "\t"
    facts_file_columns_dtypes: dict[str, Any] | None = None
    relation_column: str = "relation_id"
    subject_column: str = "subject_id"
    object_column: str = "object_id"
    timestamp_column: str | None = None
    timestamp_parser: dict | None = None
    subject_domain_column: str | None = None
    object_domain_column: str | None = None
    nodes_file: str | Path | None = None

    def get_excluded_fields(self) -> set[str]:
        excluded_files = super().get_excluded_fields()
        excluded_files.add("facts_file")
        excluded_files.add("nodes_file")
        return excluded_files
