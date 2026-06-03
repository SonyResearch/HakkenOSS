from unittest.mock import MagicMock, patch

import pytest

from data_processing.data_processor.config import DataProcessorConfig, SparkConfig
from data_processing.values import DataFrameLibrary, StorageType
from data_processing.values_config import DataFiles


def test_build_spark_returns_none_for_pandas():
    """Should not build Spark when library != PYSPARK"""
    minimal_data_files = DataFiles(relations=[])
    cfg = DataProcessorConfig(
        library=DataFrameLibrary.PANDAS,
        storage=StorageType.LOCAL,
        data_files=minimal_data_files,
        output_path="out/",
        output_sep="\t",
    )
    assert cfg.build_spark() is None
    assert cfg.spark_builder is None


def test_build_spark_raises_when_missing_spark_config():
    """Should raise if PYSPARK selected but no SparkConfig provided"""
    minimal_data_files = DataFiles(relations=[])
    cfg = DataProcessorConfig(
        library=DataFrameLibrary.PYSPARK,
        storage=StorageType.LOCAL,
        data_files=minimal_data_files,
        output_path="out/",
        output_sep="\t",
    )
    with pytest.raises(ValueError, match="Spark configuration is required"):
        cfg.build_spark()


@patch("data_processing.data_processor.config.SparkSession")
def test_build_spark_builds_builder(mock_spark):
    """Should build SparkSession.Builder and apply configs"""
    mock_builder = MagicMock()
    mock_spark.builder = mock_builder
    mock_builder.appName.return_value = mock_builder
    mock_builder.master.return_value = mock_builder
    mock_builder.config.return_value = mock_builder
    minimal_data_files = DataFiles(relations=[])

    spark_cfg = SparkConfig(
        app_name="TestApp", master="local[2]", configs={"spark.executor.memory": "2g"}
    )
    cfg = DataProcessorConfig(
        library=DataFrameLibrary.PYSPARK,
        storage=StorageType.LOCAL,
        data_files=minimal_data_files,
        output_path="out/",
        output_sep="\t",
        spark=spark_cfg,
    )

    builder = cfg.build_spark()

    # Verify that the builder is returned
    assert builder == mock_builder
    # Verify that the Spark builder was stored
    assert cfg.spark_builder == mock_builder
    # Verify that the config is applied
    mock_builder.config.assert_any_call("spark.executor.memory", "2g")
