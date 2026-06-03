from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Protocol

import pandas as pd
from ml_utils import DSVUtils
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class HypothesesDFReader(Protocol):

    @staticmethod
    def read_many(
        root_folder: Path,
        read_kwargs: Optional[Dict[str, Any]] = None,
        entity_pair_column: str | int = 1,
        relation_column: str | int = 2,
        entity_pair_regex: str = r"(\d+)\) <=====> (\d+)\)",
        score_column: str | int = 3,
    ) -> pd.DataFrame:

        df_list = []

        total_dirs = sum(1 for _ in os.walk(root_folder))

        # Then add tqdm to both loops
        for root, _dirs, files in tqdm(
            os.walk(root_folder), total=total_dirs, desc="Processing directories"
        ):
            for file in tqdm(
                files, desc=f"Processing files in {os.path.basename(root)}", leave=False
            ):
                if file.endswith(".csv"):
                    filename = Path(os.path.join(root, file))

                    df = HypothesesDFReader.read(
                        filename=filename,
                        read_kwargs=read_kwargs,
                        entity_pair_column=entity_pair_column,
                        relation_column=relation_column,
                        entity_pair_regex=entity_pair_regex,
                        score_column=score_column,
                    )
                    df["hypothesis_type"] = filename.name.replace(
                        "_inference_output_post.csv", ""
                    )

                    df_list.append(df)
        df = pd.concat(df_list, ignore_index=True)

        return df

    @staticmethod
    def read(
        filename: Path,
        read_kwargs: Optional[Dict[str, Any]] = None,
        entity_pair_column: str | int = 1,
        relation_column: str | int = 1,
        entity_pair_regex: str = r"(\d+)\) <=====> (\d+)\)",
        score_column: str | int = 3,
    ) -> pd.DataFrame:

        if read_kwargs is None:
            read_kwargs = {"delimiter": "\t"}

        try:
            logger.info(f"Loading {filename}...")
            df_raw = DSVUtils.read_dsv(filename, **read_kwargs)
            logger.info(f"Loaded raw data: {len(df_raw)} rows {df_raw.columns}")

            df_hypotheses = df_raw[entity_pair_column].str.extract(entity_pair_regex)
            df_hypotheses.rename({0: "subject", 1: "object"}, axis=1, inplace=True)
            df_hypotheses["score"] = df_raw[score_column].astype(float)
            df_hypotheses["relation"] = df_raw[relation_column].str.upper()
            df_hypotheses = df_hypotheses.dropna()
            logger.info(
                f"Processed scores data: {len(df_hypotheses)} rows after cleaning"
            )
            return df_hypotheses

        except Exception as e:
            logger.error(f"Error processing CSV data: {e!s}")
            raise
