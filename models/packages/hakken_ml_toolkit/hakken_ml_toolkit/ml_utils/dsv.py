from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from hakken_ml_toolkit.ml_utils.exceptions import (
    DelimiterExtensionMismatchError,
    StratifiedSamplingError,
)

DELIMITER_EXTENSION_MAP = {",": ".csv", "\t": ".tsv"}


class DSVUtils:
    """Delimiter Separated Values (DSV)"""

    @staticmethod
    def write_dsv(
        df: pd.DataFrame,
        file_path: Path,
        index: bool = False,
        delimiter: str = ",",
        header: bool = True,
    ) -> None:
        if (
            delimiter in DELIMITER_EXTENSION_MAP
            and DELIMITER_EXTENSION_MAP[delimiter] != file_path.suffix.lower()
        ):
            raise DelimiterExtensionMismatchError()

        logger.info(f"Saving df to {file_path}")
        df.to_csv(file_path, index=index, sep=delimiter, header=header)

    @staticmethod
    def read_dsv(  # noqa: PLR0913
        file_path: str | Path,
        dtype: defaultdict[str, Any] | None = None,
        delimiter: str | None = None,
        header: int | None = None,
        names: list[str] | None = None,
        use_cols: list[str] | None = None,
        rename_columns: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        logger.info(f"Loading DSV file: {file_path}")
        df = pd.read_csv(
            file_path,
            dtype=dtype,
            sep=delimiter,
            header=header,
            names=names,
            usecols=use_cols,
        )

        if rename_columns is not None:
            df = df.rename(columns=rename_columns)

        return df

    @staticmethod
    def read_many_dsv(
        file_path_list: list[Path],
        dtype: defaultdict[str, Any] | None = None,
        delimiter: str | None = None,
        header: int | None = None,
        names: list[str] | None = None,
    ) -> pd.DataFrame:
        df_list = []
        for file_path in file_path_list:
            df = DSVUtils.read_dsv(
                file_path=file_path,
                dtype=dtype,
                delimiter=delimiter,
                header=header,
                names=names,
            )
            df_list.append(df)

        return pd.concat(df_list, ignore_index=True)

    @staticmethod
    def stratified_sampling(
        df: pd.DataFrame,
        stratify_column: str,
        sample_fraction: float | None = None,
        sample_size: int | None = None,
        random_state: int | None = None,
    ):
        if sample_fraction is None and sample_size is None:
            msg = "Either sample_fraction or sample_size must be provided"
            raise StratifiedSamplingError(msg)

        groups = df.groupby(stratify_column, group_keys=False)

        if sample_fraction is not None:
            return groups.apply(lambda x: x.sample(frac=sample_fraction, random_state=random_state))
        if sample_size is not None:
            return groups.apply(lambda x: x.sample(n=sample_size, random_state=random_state))

        msg = "This should not happen"
        raise StratifiedSamplingError(msg)
