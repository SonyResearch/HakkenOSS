from pydantic import BaseModel, ConfigDict, Field
from pyspark.sql import SparkSession

from data_processing.utils.errors import MissingPysparkConfigurationError
from data_processing.values import DataFrameLibrary, StorageType
from data_processing.values_config import DataFiles


class SparkConfig(BaseModel):
    app_name: str = "MyApp"
    master: str = "local[*]"
    configs: dict[str, str] = Field(default_factory=dict)


class DataProcessorConfig(BaseModel):
    library: DataFrameLibrary = DataFrameLibrary.PANDAS
    storage: StorageType = StorageType.LOCAL
    data_files: DataFiles
    output_path: str
    output_sep: str
    spark: SparkConfig | None = None
    spark_builder: SparkSession.Builder | None = None

    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    def build_spark(self):
        """Build SparkSession.Builder from SparkConfig if applicable."""
        if self.library != DataFrameLibrary.PYSPARK:
            return None

        if not self.spark:
            raise MissingPysparkConfigurationError

        builder = SparkSession.builder.appName(self.spark.app_name).master(self.spark.master)
        for key, value in self.spark.configs.items():
            builder = builder.config(key, value)

        self.spark_builder = builder
        return builder
