import unittest

from data_processing.adapters.impl.pandas_adapter import PandasAdapter
from data_processing.adapters.impl.pyspark_adapter import PySparkAdapter
from data_processing.factories.adapter_factory import LibraryAdapterFactory
from data_processing.utils.errors import UnsupportedLibraryError
from data_processing.values import DataFrameLibrary


class TestAdapterFactory(unittest.TestCase):
    """Test AdapterFactory"""

    def test_processors_dict_creation(self):
        """Test that the adapters dict is created as private members
        at instantiation"""
        result = LibraryAdapterFactory._adapters

        available_adapters = {
            DataFrameLibrary.PANDAS: PandasAdapter,
            DataFrameLibrary.PYSPARK: PySparkAdapter,
        }

        # Verify dicts are equal
        self.assertDictEqual(result, available_adapters)

    def test_get_adapters(self):
        """Test that the correct adapter is returned"""
        # Pandas
        pandas_library = DataFrameLibrary.PANDAS
        result = LibraryAdapterFactory.get_adapter(pandas_library)
        self.assertIs(result, PandasAdapter)

        # Pyspark
        pyspark_library = DataFrameLibrary.PYSPARK
        result = LibraryAdapterFactory.get_adapter(pyspark_library)
        self.assertIs(result, PySparkAdapter)

    def test_get_non_available_adapter(self):
        """Test that the correct adapter is returned"""
        # Polar's adapter not yet implemented
        polars_library = DataFrameLibrary.POLARS
        with self.assertRaises(UnsupportedLibraryError):
            LibraryAdapterFactory.get_adapter(polars_library)
