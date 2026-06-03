from typing import ClassVar

from data_processing.adapters.impl.pandas_adapter import PandasAdapter
from data_processing.adapters.impl.pyspark_adapter import PySparkAdapter
from data_processing.utils.errors import UnsupportedLibraryError
from data_processing.values import DataFrameLibrary


class LibraryAdapterFactory:
    """Factory to get the appropriate library adapter"""

    _adapters: ClassVar[dict] = {
        DataFrameLibrary.PANDAS: PandasAdapter,
        DataFrameLibrary.PYSPARK: PySparkAdapter,
    }

    @classmethod
    def get_adapter(cls, library: DataFrameLibrary):
        if library not in cls._adapters:
            supported_libraries = [lib.name for lib in cls._adapters]
            raise UnsupportedLibraryError(supported_libraries)
        return cls._adapters.get(library)
