import unittest
from unittest.mock import Mock

from data_processing.data_processor.impl.processor_digital_science import DigitalScienceProcessor
from data_processing.data_processor.impl.processor_dummy import DummyProcessor
from data_processing.data_processor.impl.processor_pubtator import PubtatorProcessor
from data_processing.factories.processor_factory import DatasetProcessorFactory
from data_processing.utils.errors import UnknownDatasetError
from data_processing.values import DataFrameLibrary


class TestProcessorFactory(unittest.TestCase):
    """Test AdapterFactory"""

    def test_processors_dict_creation(self):
        """Test that the processors dict is created as private members
        at instantiation"""
        result = DatasetProcessorFactory._processors

        available_processors = {
            "dummy": DummyProcessor,
            "pubtator": PubtatorProcessor,
            "digital_science": DigitalScienceProcessor,
        }

        # Verify dicts are equal
        self.assertDictEqual(result, available_processors)

    def test_get_processors(self):
        """Test that the correct processor is returned"""
        config = Mock()
        config.library = DataFrameLibrary.PANDAS

        # Dummy
        dataset_name = "dummy"
        result = DatasetProcessorFactory.create_processor(dataset_name, config)
        self.assertIsInstance(result, DummyProcessor)

        # Pubtator
        dataset_name = "pubtator"
        result = DatasetProcessorFactory.create_processor(dataset_name, config)
        self.assertIsInstance(result, PubtatorProcessor)

    def test_get_non_available_processor(self):
        """Test that the correct processor is returned"""
        # Polar's adapter not yet implemented
        config = Mock()
        config.library = DataFrameLibrary.PANDAS

        unavailable_dataset = "non_available_dataset"
        with self.assertRaises(UnknownDatasetError):
            DatasetProcessorFactory.create_processor(unavailable_dataset, config)
