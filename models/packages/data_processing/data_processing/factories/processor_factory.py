from typing import Any, ClassVar

from data_processing.data_processor.config import DataProcessorConfig
from data_processing.data_processor.impl.processor_digital_science import DigitalScienceProcessor
from data_processing.data_processor.impl.processor_dummy import DummyProcessor
from data_processing.data_processor.impl.processor_pubtator import PubtatorProcessor
from data_processing.data_processor.processor_base import DataProcessor
from data_processing.utils.errors import UnknownDatasetError


class DatasetProcessorFactory:
    """Factory to create the appropriate dataset processor"""

    _processors: ClassVar[dict[str, type[DataProcessor[Any]]]] = {
        "dummy": DummyProcessor,
        "pubtator": PubtatorProcessor,
        "digital_science": DigitalScienceProcessor,
    }

    @classmethod
    def create_processor(cls, dataset_name: str, config: DataProcessorConfig) -> DataProcessor:
        processor_class = cls._processors.get(dataset_name.lower())
        if not processor_class:
            raise UnknownDatasetError(dataset_name)

        return processor_class(config)
