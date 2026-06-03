from __future__ import annotations

import json
from typing import TYPE_CHECKING, ClassVar

from hakken_ml_toolkit.ml_utils import DSVUtils

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd


class DigitalScienceUtils:
    MISSING: ClassVar = "UNKNOWN"
    COLUMN_DTYPES: ClassVar = {
        "ocid_subject": str,
        "ocid_object": str,
        "ocid_relation": str,
    }

    @staticmethod
    def load_relation_filter(jsonl_path: Path, format_relations: bool = True) -> list[str]:
        relation_map_list = []
        with open(jsonl_path, encoding="utf-8") as file:
            for line in file:
                if line.strip():  # Skip empty lines
                    relation_map_list.append(json.loads(line))

        def format_relation(relation: str) -> str:
            if format_relations:
                return relation.upper().replace(" ", "_").strip()
            return relation

        relation_filter = set()
        for relation_map in relation_map_list:
            if relation_map["action"] == "KEEP":
                if relation_map["relation_new"]:
                    relation_filter.add(format_relation(relation_map["relation_new"]))
                else:
                    relation_filter.add(format_relation(relation_map["relation_original"]))

            if relation_map["action"] == "SWAP":
                relation_filter.add(format_relation(relation_map["relation_new"]))

        return list(relation_filter)

    @staticmethod
    def load_filtered_edges_df(
        edges_file: str | Path,
        relation_filter: list[str] | None = None,
        relation_column: str = "relation",
    ) -> pd.DataFrame:
        edges_df: pd.DataFrame = DSVUtils.read_dsv(
            edges_file,
            delimiter="\t",
            header=0,
            dtype=DigitalScienceUtils.COLUMN_DTYPES,
        )

        if relation_filter is not None:
            edges_df = edges_df[edges_df[relation_column].isin(relation_filter)]
        return edges_df

    @staticmethod
    def migrate_relations_v01_to_v02(relation_map_list: list[dict]) -> list[dict]:
        """
        This function processes a list of relation mappings from version 0.1 and adjusts
        them to be compatible with version 0.2 requirements. It evaluates each relation
        and assigns appropriate actions (KEEP, SWAP) and transforms relation types when needed.

        For further information, please contact Alessandra Toniato.

        Args:
            relation_map_list: A list of dictionaries containing relation mappings
                            from version 0.1

        Returns:
            A modified copy of the input list with updated relation mappings for version 0.2
        """

        relation_map_output = relation_map_list.copy()
        for relation_map in relation_map_output:
            if relation_map["relation_original"] == "affects transcriptional activity of":
                relation_map["action"] = "KEEP"
            if relation_map["relation_original"] == "affects degradation of":
                relation_map["action"] = "KEEP"
            if relation_map["relation_original"] == "of":
                relation_map["action"] = "SWAP"
                relation_map["relation_new"] = "is a biomarker"

        return relation_map_output
