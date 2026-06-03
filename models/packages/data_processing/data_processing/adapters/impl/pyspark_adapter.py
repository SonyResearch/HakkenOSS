from collections.abc import Callable
from functools import reduce
from pathlib import Path
from typing import Any, get_type_hints

from loguru import logger
from pyspark.sql import Column, SparkSession
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F  # noqa: N812
from pyspark.sql.types import BooleanType, DataType, FloatType, IntegerType, StringType

from data_processing.adapters.adapter_base import AdapterBase
from data_processing.utils.errors import (
    ColumnCountMismatchError,
    EmptyListError,
    MissingColumnError,
    MissingPysparkBuilderError,
    UnsupportedAggregationFunctionError,
)
from data_processing.utils.pyspark_utils import write_to_single_dsv_file
from data_processing.values import DEFAULT_SEPARATOR, PROCESSOR_CACHE_PATH, StorageType


class PySparkAdapter(AdapterBase[SparkDataFrame]):
    """Adapter for PySpark operations"""

    @staticmethod
    def read_csv(path: str, **kwargs) -> SparkDataFrame:
        # Create a spark session
        spark_builder = kwargs.get("spark_builder")
        if spark_builder is None:
            raise MissingPysparkBuilderError
        spark_session = spark_builder.getOrCreate()

        # Replace s3:// with s3a:// for S3 storage
        storage = kwargs.get("storage", StorageType.LOCAL)
        path = path.replace("s3://", "s3a://") if storage == StorageType.S3 else path

        # Extract parameters from kwargs with defaults
        sep = kwargs.get("sep", DEFAULT_SEPARATOR)
        header = kwargs.get("header", True)
        schema = kwargs.get("schema")
        infer_schema = kwargs.get("inferSchema", True)
        repartition = kwargs.get("repartition", 200)

        # Read CSV
        df: SparkDataFrame = spark_session.read.csv(
            path, sep=sep, header=header, schema=schema, inferSchema=infer_schema
        )
        column_names = kwargs.get("column_names")
        if column_names:
            if len(df.columns) != len(column_names):
                raise ColumnCountMismatchError(len(df.columns), len(column_names))
            df = df.toDF(*column_names)

        # Repartition if specified
        if repartition:
            df = df.repartition(repartition)

        return df

    @staticmethod
    def concat(list_of_dataframes: list[SparkDataFrame]) -> SparkDataFrame:
        if len(list_of_dataframes) == 0:
            raise EmptyListError

        final_dataframe = list_of_dataframes[0]
        if len(list_of_dataframes) > 1:
            for df in list_of_dataframes[1:]:
                final_dataframe = final_dataframe.unionByName(df, allowMissingColumns=True)
        return final_dataframe

    @staticmethod
    def to_csv(df: SparkDataFrame, path: str, **kwargs) -> None:
        # Extract parameters from kwargs with defaults
        sep = kwargs.get("output_sep", DEFAULT_SEPARATOR)
        header = kwargs.get("header", True)
        mode = kwargs.get("mode", "overwrite")
        write_single_file = kwargs.get("write_single_file", True)

        # Write CSV
        df.write.mode(mode).csv(path, sep=sep, header=header)

        # Optionally consolidate to single file
        if write_single_file:
            write_to_single_dsv_file(path, sep=sep)

    @staticmethod
    def dropna(df: SparkDataFrame, **kwargs) -> SparkDataFrame:
        return df.dropna(**kwargs)

    @staticmethod
    def drop_duplicates(df: SparkDataFrame, **kwargs) -> SparkDataFrame:
        return df.dropDuplicates(**kwargs)

    @staticmethod
    def replace_pattern(
        df: SparkDataFrame, columns: list[str] | str, pattern: str, replacement: str = ""
    ) -> SparkDataFrame:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise MissingColumnError(", ".join(missing))
        for col in columns:
            df = df.withColumn(col, F.regexp_replace(col, pattern, replacement))
        return df

    @staticmethod
    def show_head(df: SparkDataFrame, n: int = 10) -> None:
        head_df = df.limit(n).toPandas()
        logger.info(f"Dataframe sample:\n{head_df}")

    @staticmethod
    def count_rows(df: SparkDataFrame) -> int:
        return df.count()

    @staticmethod
    def split_column(
        df: SparkDataFrame,
        column_name: str,
        column1_name: str,
        column2_name: str,
        separator: str = "\\|",
    ) -> SparkDataFrame:
        df = df.withColumn(column1_name, F.split(F.col(column_name), separator).getItem(0))
        df = df.withColumn(column2_name, F.split(F.col(column_name), separator).getItem(1))
        return df  # noqa: RET504

    @staticmethod
    def concatenate_columns(
        df: SparkDataFrame, columns: list[str], new_column: str, delimiter: str = "|"
    ) -> SparkDataFrame:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise MissingColumnError(", ".join(missing))
        return df.withColumn(new_column, F.concat_ws(delimiter, *[F.col(c) for c in columns]))

    @staticmethod
    def filter_by_value(
        df: SparkDataFrame, columns: list[str], invalid_value: str = "-"
    ) -> SparkDataFrame:
        # Build condition: all columns != invalid_value
        condition: Column = reduce(
            lambda acc, col: acc & (F.col(col) != invalid_value),
            columns[1:],
            F.col(columns[0]) != invalid_value,
        )

        return df.filter(condition)

    @staticmethod
    def filter_by_condition(df: SparkDataFrame, condition: str) -> SparkDataFrame:
        return df.filter(F.expr(condition))

    @staticmethod
    def apply_function_to_column(
        df: SparkDataFrame, column: str, new_column: str, func: Callable
    ) -> SparkDataFrame:
        # --- Infer return type from the callable ---
        type_hints = get_type_hints(func)
        return_type = type_hints.get("return", str)

        # --- Map Python type → Spark SQL type ---
        spark_type_map: dict[type, DataType] = {
            str: StringType(),
            int: IntegerType(),
            float: FloatType(),
            bool: BooleanType(),
        }

        if return_type in spark_type_map:
            spark_return_type = spark_type_map[return_type]
        else:
            logger.warning(
                f"Unsupported return type {return_type.__name__}, defaulting to StringType."
            )
            spark_return_type = StringType()

        # --- Create UDF and apply ---
        udf_wrapper = F.udf(func, spark_return_type)
        return df.withColumn(new_column, udf_wrapper(F.col(column)))

    @staticmethod
    def select_columns(df: SparkDataFrame, columns: list[str]) -> SparkDataFrame:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise MissingColumnError(", ".join(missing))
        return df.select(*columns)

    @staticmethod
    def drop_columns(df: SparkDataFrame, columns: list[str]) -> SparkDataFrame:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise MissingColumnError(", ".join(missing))
        return df.drop(*columns)

    @staticmethod
    def column_has_nulls(df: SparkDataFrame, column: str) -> bool:
        if column not in df.columns:
            raise MissingColumnError(column)
        return bool(df.filter(F.col(column).isNull()).limit(1).count() > 0)

    @staticmethod
    def unique_values(df: SparkDataFrame, column: str) -> list[Any]:
        """
        Return a list of unique (distinct) values in the specified column.

        Args:
            df (SparkDataFrame): Input DataFrame.
            column (str): Column name to extract unique values from.

        Returns:
            list: Python list of unique values in the column.

        Raises:
            MissingColumnError: If the specified column is not present in the DataFrame.
        """
        if column not in df.columns:
            raise MissingColumnError(column)

        return [row[column] for row in df.select(column).distinct().collect()]

    @staticmethod
    def groupby_aggregate(
        df: SparkDataFrame,
        groupby_cols: list[str],
        aggregations: dict[str, str | list[str]],
    ) -> SparkDataFrame:
        agg_exprs = []

        # Handle both single and multiple aggregations per column
        for col_name, agg_func in aggregations.items():
            if isinstance(agg_func, str):
                agg_func_name = "collect_list" if agg_func == "list" else agg_func
                agg_exprs.append(
                    getattr(F, agg_func_name)(F.col(col_name)).alias(f"{col_name}_{agg_func}")
                )
            elif isinstance(agg_func, list):
                for func_name in agg_func:
                    agg_func_name = "collect_list" if func_name == "list" else func_name
                    agg_exprs.append(
                        getattr(F, agg_func_name)(F.col(col_name)).alias(f"{col_name}_{func_name}")
                    )
            else:
                raise UnsupportedAggregationFunctionError(col_name, agg_func)

        return df.groupBy(*groupby_cols).agg(*agg_exprs)

    @staticmethod
    def union(df1: SparkDataFrame, df2: SparkDataFrame) -> SparkDataFrame:
        # unionByName allows flexible column order
        return df1.unionByName(df2, allowMissingColumns=True).distinct()

    @staticmethod
    def rename_columns(df: SparkDataFrame, rename_map: dict[str, str]) -> SparkDataFrame:
        # sequentially alias columns
        result = df
        for old, new in rename_map.items():
            result = result.withColumnRenamed(old, new)
        return result

    @staticmethod
    def materialize_data(
        df: SparkDataFrame, name: str, timestamp: str | None = None
    ) -> SparkDataFrame:
        output_folder = (
            Path(PROCESSOR_CACHE_PATH) / timestamp / name
            if timestamp is not None
            else Path(PROCESSOR_CACHE_PATH) / name
        )
        spark_session = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
        df.write.mode("overwrite").parquet(str(output_folder))
        return spark_session.read.parquet(str(output_folder))
