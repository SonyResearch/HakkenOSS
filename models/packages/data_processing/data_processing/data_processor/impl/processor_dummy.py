from datetime import datetime
from pathlib import Path

from data_processing.data_processor.processor_base import DataProcessor
from data_processing.values import OUTPUT_RELATIONS_FILENAME, DataFrameType


class DummyProcessor(DataProcessor[DataFrameType]):
    """Processor for Pubtator dataset with specific cleaning strategies"""

    def process(self) -> DataFrameType:
        """Pubtator-specific cleaning strategies"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        relations_file = self.config.data_files.relations[0]

        # Load data into the dataframe self.df
        self.df = self.load_data(relations_file)

        # Strategy 1: Remove rows with missing critical fields
        self.df = self.adapter.dropna(self.df, subset=["pmid", "subject_id"])
        self.df = self.adapter.materialize_data(self.df, "step1", timestamp)

        # Strategy 2: Remove duplicates based on PMID
        self.df = self.adapter.drop_duplicates(self.df, subset=["pmid"])
        self.df = self.adapter.materialize_data(self.df, "step2", timestamp)

        path = Path(self.config.output_path) / OUTPUT_RELATIONS_FILENAME
        self.save_data(str(path))

        return self.df
