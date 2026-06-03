class UnknownDatasetError(ValueError):
    """Raised when an unknown dataset name is provided"""

    def __init__(self, dataset_name: str):
        super().__init__(f"Unknown dataset: {dataset_name}")


class UnsupportedLibraryError(ValueError):
    """Raised when an unsupported library is chosen"""

    def __init__(self, supported_libraries: list[str]):
        super().__init__(
            "The requested dataframe library is not supported."
            f"Supported libraries: {supported_libraries}"
        )


class MissingPysparkBuilderError(ValueError):
    """Raised when a pyspark builder is not provided"""

    def __init__(self):
        super().__init__("Missing Pyspark Builder!")


class MissingPysparkConfigurationError(ValueError):
    """Raised when a pyspark builder is not provided"""

    def __init__(self):
        super().__init__("Spark configuration is required for PySpark library.")


class MissingColumnError(KeyError):
    """Raised when a column is not found in a Dataframe"""

    def __init__(self, column_name: str):
        super().__init__(f"Column '{column_name}' not available in dataframe.")


class UnsupportedAggregationFunctionError(TypeError):
    """Raised when an unknown aggregation function for groupby is provided"""

    def __init__(self, column_name: str, agg_func: str):
        super().__init__(
            f"Unsupported aggregation type for column '{column_name}': {type(agg_func)}"
        )


class ColumnContainsNullError(Exception):
    """Raised when a DataFrame column contains None, NaN, or null values."""

    def __init__(self, column: str):
        super().__init__(f"Column '{column}' contains null values!")


class EmptyListError(Exception):
    """Raised when a list is empty."""

    def __init__(self):
        super().__init__("A non empty list is required!")


class ColumnCountMismatchError(KeyError):
    """Raised when in a Dataframe trying to assign more column names than columns available"""

    def __init__(self, len_expected_cols: int, len_provided_cols: int):
        super().__init__(
            f"Column count mismatch: expected {len_expected_cols}, got {len_provided_cols} ."
        )
