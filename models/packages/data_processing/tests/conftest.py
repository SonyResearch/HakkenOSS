# conftest.py
import importlib.util
from pathlib import Path

PYSPARK_TEST_FILES = [
    "test_pyspark_adapter.py",
    "test_pyspark_utils.py",
    "test_adapter_factory.py",
    "test_config_creation.py",
    "test_pandas_adapter.py",
    "test_processor_factory.py",
    "test_processor_pubtator.py",
    "test_values.py",
]


def pytest_ignore_collect(collection_path: Path | str) -> bool | None:
    """Skip test files that require Spark if Spark is not installed."""
    spark_available = importlib.util.find_spec("pyspark") is not None

    if not spark_available:
        # Convert to Path if it's a string
        if isinstance(collection_path, str):
            collection_path = Path(collection_path)

        # Skip if the file is in the list of PySpark test files
        if (
            collection_path.is_file()
            and collection_path.suffix == ".py"
            and collection_path.name in PYSPARK_TEST_FILES
        ):
            return True
    return None
