from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pandas as pd
from loguru import logger

from data_processing.adapters.adapter_base import AdapterBase
from data_processing.utils.errors import (
    ColumnCountMismatchError,
    EmptyListError,
    MissingColumnError,
)
from data_processing.values import (
    DEFAULT_SEPARATOR,
    PROCESSOR_CACHE_PATH,
    SEPARATOR_SUFFIX_DICT,
    UNKNOWN_SUFFIX,
)


class PandasAdapter(AdapterBase[pd.DataFrame]):
    """Adapter for pandas operations"""

    @staticmethod
    def read_csv(path: str, **kwargs) -> pd.DataFrame:
        header = kwargs.get("header", 0)
        sep = kwargs.get("sep", DEFAULT_SEPARATOR)
        column_names = kwargs.get("column_names")

        df = pd.read_csv(path, sep=sep, header=header)

        if column_names:
            if df.shape[1] != len(column_names):
                raise ColumnCountMismatchError(df.shape[1], len(column_names))
            df.columns = column_names

        return df

    @staticmethod
    def to_csv(df: pd.DataFrame, path: str, **kwargs) -> None:
        sep = kwargs.get("output_sep", DEFAULT_SEPARATOR)
        index = kwargs.get("index", False)

        filepath_with_suffix = f"{path}{SEPARATOR_SUFFIX_DICT.get(sep, UNKNOWN_SUFFIX)}"
        output_path = Path(filepath_with_suffix)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path, sep=sep, index=index)

    @staticmethod
    def concat(list_of_dataframes: list[pd.DataFrame]) -> pd.DataFrame:
        if len(list_of_dataframes) == 0:
            raise EmptyListError
        return pd.concat(list_of_dataframes)

    @staticmethod
    def dropna(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        result: pd.DataFrame = df.dropna(**kwargs)

        return result

    @staticmethod
    def drop_duplicates(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        result: pd.DataFrame = df.drop_duplicates(**kwargs)

        return result

    @staticmethod
    def replace_pattern(
        df: pd.DataFrame, columns: list[str] | str, pattern: str, replacement: str = ""
    ) -> pd.DataFrame:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise MissingColumnError(", ".join(missing))
        for col in columns:
            df[col] = df[col].astype(str).str.replace(pattern, replacement, regex=True)

        return df

    @staticmethod
    def show_head(df: pd.DataFrame, n: int = 10) -> None:
        head_df = df.head(n)
        logger.info(f"Dataframe sample:\n{head_df}")

    @staticmethod
    def count_rows(df: pd.DataFrame) -> int:
        count = len(df)
        logger.info(f"Number of rows in DataFrame: {count}")

        return count

    @staticmethod
    def split_column(
        df: pd.DataFrame,
        column_name: str,
        column1_name: str,
        column2_name: str,
        separator: str = "\\|",
    ) -> pd.DataFrame:
        split_cols = df[column_name].str.split(separator, n=1, expand=True)
        df[column1_name] = split_cols[0]
        df[column2_name] = split_cols[1]
        logger.info(
            f"Split column '{column_name}' into '{column1_name}' and '{column2_name}'"
            f" using separator '{separator}'"
        )

        return df

    @staticmethod
    def concatenate_columns(
        df: pd.DataFrame, columns: list[str], new_column: str, delimiter: str = "|"
    ) -> pd.DataFrame:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise MissingColumnError(", ".join(missing))
        df = df.copy()
        df[new_column] = df[columns].astype(str).agg(delimiter.join, axis=1)
        return df

    @staticmethod
    def filter_by_value(
        df: pd.DataFrame, columns: list[str], invalid_value: str = "-"
    ) -> pd.DataFrame:
        mask = (df[columns] != invalid_value).all(axis=1)
        df_filtered = df[mask]
        logger.info(f"Filtered rows where any of the columns {columns} were '{invalid_value}'")

        return df_filtered

    @staticmethod
    def filter_by_condition(df: pd.DataFrame, condition: str) -> pd.DataFrame:
        df_filtered = df.query(condition).copy()
        logger.info(f"Filtered rows with condition: {condition}")
        return df_filtered

    @staticmethod
    def apply_function_to_column(
        df: pd.DataFrame, column: str, new_column: str, func: Callable
    ) -> pd.DataFrame:
        df[new_column] = df[column].apply(func)
        return df

    @staticmethod
    def select_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise MissingColumnError(", ".join(missing))
        return df[columns].copy()

    @staticmethod
    def drop_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise MissingColumnError(", ".join(missing))
        return df.drop(columns=columns)

    @staticmethod
    def column_has_nulls(df: pd.DataFrame, column: str) -> bool:
        if column not in df.columns:
            raise MissingColumnError(column)
        return bool(df[column].isna().any())

    @staticmethod
    def unique_values(df: pd.DataFrame, column: str) -> list[Any]:
        if column not in df.columns:
            raise MissingColumnError(column)

        return cast("list[Any]", df[column].dropna().unique().tolist())

    @staticmethod
    def groupby_aggregate(
        df: pd.DataFrame,
        groupby_cols: list[str],
        aggregations: dict[str, str | list[str]],
    ) -> pd.DataFrame:
        # Convert 'list' string to actual Python list function
        agg_map = {}
        for col, funcs in aggregations.items():
            if isinstance(funcs, str):
                agg_map[col] = list if funcs == "list" else funcs
            elif isinstance(funcs, list):
                agg_map[col] = [list if f == "list" else f for f in funcs]

        # Apply groupby and aggregation
        grouped = df.groupby(groupby_cols).agg(agg_map).reset_index()  # type: ignore

        # Flatten MultiIndex if needed
        if isinstance(grouped.columns, pd.MultiIndex):
            grouped.columns = [
                "_".join([str(c) for c in col if c]) for col in grouped.columns.to_flat_index()
            ]
        else:
            # For single aggregation, rename column to include aggregation
            for col, agg in aggregations.items():
                if isinstance(agg, str):
                    grouped.rename(columns={col: f"{col}_{agg}"}, inplace=True)

        return cast("pd.DataFrame", grouped)

    @staticmethod
    def union(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        return pd.concat([df1, df2], ignore_index=True).drop_duplicates(ignore_index=True)

    @staticmethod
    def rename_columns(df: pd.DataFrame, rename_map: dict[str, str]) -> pd.DataFrame:
        return df.rename(columns=rename_map, copy=True)

    @staticmethod
    def materialize_data(df: pd.DataFrame, name: str, timestamp: str | None = None) -> pd.DataFrame:
        output_file = (
            Path(PROCESSOR_CACHE_PATH) / timestamp / f"{name}.parquet"
            if timestamp is not None
            else Path(PROCESSOR_CACHE_PATH) / f"{name}.parquet"
        )
        output_file.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        df.to_parquet(output_file, engine="pyarrow", index=False)

        return pd.read_parquet(output_file, engine="pyarrow")
