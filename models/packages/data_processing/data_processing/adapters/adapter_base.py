from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Generic

from data_processing.values import DataFrameType


class AdapterBase(ABC, Generic[DataFrameType]):
    """Abstract base class for the libraries adapters"""

    @staticmethod
    @abstractmethod
    def read_csv(path: str, **kwargs) -> DataFrameType:
        pass

    @staticmethod
    @abstractmethod
    def to_csv(df: DataFrameType, path: str, **kwargs) -> None:
        pass

    @staticmethod
    @abstractmethod
    def concat(list_of_dataframes: list[DataFrameType]) -> DataFrameType:
        pass

    @staticmethod
    @abstractmethod
    def dropna(df: DataFrameType, **kwargs) -> DataFrameType:
        """
        Drop rows with null values

        Args:
            df: Spark DataFrame
            **kwargs: Parameters for dropna (e.g., subset, how, thresh)

        Returns:
            DataFrame with null values dropped
        """

        pass

    @staticmethod
    @abstractmethod
    def drop_duplicates(df: DataFrameType, **kwargs) -> DataFrameType:
        """
        Drop duplicate rows

        Args:
            df: Spark DataFrame
            **kwargs: Parameters for dropDuplicates (e.g., subset)

        Returns:
            DataFrame with duplicates removed
        """
        pass

    @staticmethod
    @abstractmethod
    def replace_pattern(
        df: DataFrameType, columns: list[str] | str, pattern: str, replacement: str = ""
    ) -> DataFrameType:
        pass

    @staticmethod
    @abstractmethod
    def show_head(df: DataFrameType, n: int = 10) -> None:
        pass

    @staticmethod
    @abstractmethod
    def count_rows(df: DataFrameType) -> int:
        """
        Returns the number of rows in the given pandas DataFrame.
        """
        pass

    @staticmethod
    @abstractmethod
    def split_column(
        df: DataFrameType,
        column_name: str,
        column1_name: str,
        column2_name: str,
        separator: str = "\\|",
    ) -> DataFrameType:
        """
        Split a column into two new columns based on a separator.

        Example:
            input column "domain_pipe_subject_id_raw" = "Chemical|MESH:C027078"
            → column1_name='Domain', column2_name='subject_id_raw'
        """
        pass

    @staticmethod
    @abstractmethod
    def concatenate_columns(
        df: DataFrameType, columns: list[str], new_column: str, delimiter: str = "|"
    ) -> DataFrameType:
        """
        Concatenate multiple columns into a new column with a given delimiter.

        Args:
            df: DataFrame (either pandas or Spark)
            columns: list of column names to concatenate
            new_column: name of the resulting column
            delimiter: string used to join values (default: "|")
        """
        pass

    @staticmethod
    @abstractmethod
    def filter_by_value(
        df: DataFrameType, columns: list[str], invalid_value: str = "-"
    ) -> DataFrameType:
        """
        Remove rows where any of the specified columns has the invalid_value.
        """
        pass

    @staticmethod
    @abstractmethod
    def filter_by_condition(df: DataFrameType, condition: str) -> DataFrameType:
        """
        Keep only rows that satisfy the provided condition.
        Example: "col_name_1 > 1 or col_name_2 > 1"
        """
        pass

    @staticmethod
    @abstractmethod
    def apply_function_to_column(
        df: DataFrameType, column: str, new_column: str, func: Callable[[str], str]
    ) -> DataFrameType:
        """
        Apply a function to a column and create a new column with transformed value.

        Args:
            df: pandas DataFrame
            column: column to apply function to
            new_column: name of the resulting column
            func: function that takes a string and returns a string
        """
        pass

    @staticmethod
    @abstractmethod
    def materialize_data(
        df: DataFrameType, name: str, timestamp: str | None = None
    ) -> DataFrameType:
        """
        Saves intermediate files with the data

        Args:
            df: Spark DataFrame
            name: A name to tag the files saved
            timestamp: if provided creates a cache folder with timestamp

        Returns:
            The same dataframe that was saved
        """
        pass

    @staticmethod
    @abstractmethod
    def select_columns(df: DataFrameType, columns: list[str]) -> DataFrameType:
        """
        Select a subset of columns from the dataframe.
        """
        pass

    @staticmethod
    @abstractmethod
    def drop_columns(df: DataFrameType, columns: list[str]) -> DataFrameType:
        """
        Drops a selection of columns from the dataframe
        """
        pass

    @staticmethod
    @abstractmethod
    def column_has_nulls(df: DataFrameType, column: str) -> bool:
        """Return True if the column contains any null values."""
        pass

    @staticmethod
    @abstractmethod
    def unique_values(df: DataFrameType, column: str) -> list:
        """
        Return a list of unique (distinct) values in the specified column.
        """
        pass

    @staticmethod
    @abstractmethod
    def groupby_aggregate(
        df: DataFrameType,
        groupby_cols: list[str],
        aggregations: dict[str, str | list[str]],
    ) -> DataFrameType:
        """
        Perform groupby and aggregation operations.
        Example of `aggregations`: {"col1": "count", "col2": ["nunique", "sum"]}
        """
        pass

    @staticmethod
    @abstractmethod
    def union(df1: DataFrameType, df2: DataFrameType) -> DataFrameType:
        """
        Perform a union of two dataframes (row-wise concatenation).
        Should remove duplicate columns automatically if necessary.
        """
        pass

    @staticmethod
    @abstractmethod
    def rename_columns(df: DataFrameType, rename_map: dict[str, str]) -> DataFrameType:
        """
        Rename columns using rename_map {old_name: new_name}.
        """
        pass
