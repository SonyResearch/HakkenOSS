from abc import ABC, abstractmethod
from typing import Generic

from data_processing.data_processor.config import DataProcessorConfig
from data_processing.factories.adapter_factory import LibraryAdapterFactory
from data_processing.values import DataFrameLibrary, DataFrameType
from data_processing.values_config import BaseFileConfig


class DataProcessor(ABC, Generic[DataFrameType]):
    """Abstract base class for data processors"""

    def __init__(self, config: DataProcessorConfig):
        self.config = config
        self.df: DataFrameType | None = None
        self.df_ontologies: DataFrameType | None = None
        self.library = self.config.library
        self.adapter = LibraryAdapterFactory.get_adapter(self.library)

        if self.library == DataFrameLibrary.PYSPARK:
            self.config.build_spark()

    def load_data(self, file_cfg: BaseFileConfig) -> DataFrameType:
        """
        Loads a dataset using the chosen adapter and per-file configuration.
        """
        kwargs = {
            "sep": file_cfg.sep,
            "header": file_cfg.header,
            "column_names": file_cfg.column_names,
            "encoding": file_cfg.encoding,
            "spark_builder": self.config.spark_builder,
            # optionally pass more things your adapter supports
        }

        df: DataFrameType = self.adapter.read_csv(str(file_cfg.path), **kwargs)
        return df

    @abstractmethod
    def process(self) -> DataFrameType:
        """Process the data with specific cleaning strategies"""
        pass

    def save_data(self, path: str) -> None:
        """Saves the processed dataframe to file"""
        self.adapter.to_csv(self.df, path, **self.config.dict())

    def get_data(self) -> DataFrameType | None:
        """Return the processed dataframe"""
        return self.df
