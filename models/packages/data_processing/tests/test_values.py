import unittest

from data_processing.values import DataFrameLibrary, StorageType


class TestDataFrameLibraryEnum(unittest.TestCase):
    """Test DataFrameLibrary enum"""

    def test_data_frame_library_types_exist(self):
        """Verify DataFrameLibrary enum has expected values"""
        self.assertTrue(hasattr(DataFrameLibrary, "PANDAS"))
        self.assertTrue(hasattr(DataFrameLibrary, "PYSPARK"))
        self.assertTrue(hasattr(DataFrameLibrary, "POLARS"))


class TestStorageTypeEnum(unittest.TestCase):
    """Test StorageType enum"""

    def test_storage_types_exist(self):
        """Verify StorageType enum has expected values"""
        self.assertTrue(hasattr(StorageType, "LOCAL"))
        self.assertTrue(hasattr(StorageType, "S3"))
