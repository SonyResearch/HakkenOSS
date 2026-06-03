from enum import Enum
from typing import TypeVar

import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame

# Define a generic DataFrame type
DataFrameType = TypeVar("DataFrameType", bound=pd.DataFrame | SparkDataFrame)


class DataFrameLibrary(Enum):
    """Enum for supported DataFrame libraries"""

    PANDAS = "pandas"
    PYSPARK = "pyspark"
    POLARS = "polars"


class StorageType(Enum):
    S3 = "s3"
    LOCAL = "local"


DEFAULT_SEPARATOR = "\t"

SEPARATOR_SUFFIX_DICT = {"\t": ".tsv", ",": ".csv"}

PROCESSOR_CACHE_PATH = "artifacts"

OUTPUT_RELATIONS_FILENAME = "edges"

UNKNOWN_SUFFIX = ".unknown"


# default columns for data processing
LIST_TO_STRING_SEPARATOR = "|"
PMID_COLUMN = "pmid"
PMIDS_COLUMN = "pmids"  # used when removing time duplicates of a triple
RELATION_TYPE_COLUMN = "relation_type"
RELATION_ID_COLUMN = "relation_id"
NODE_DOMAIN_COLUMN = "node_domain"
NODE_ID_COLUMN = "node_id"
NODE_ID_RAW_COLUMN = "node_id_raw"
SUBJECT_DOMAIN_COLUMN = "subject_domain"
SUBJECT_ID_COLUMN = "subject_id"
SUBJECT_ID_RAW_COLUMN = "subject_id_raw"
OBJECT_DOMAIN_COLUMN = "object_domain"
OBJECT_ID_COLUMN = "object_id"
TIMESTAMP_COLUMN = "year"
LICENCE_COLUMN = "licence"
NUMBER_OF_OCCURRENCES_COLUMN = "number_of_occurrences"
YEAR_OCCURRENCES_COLUMN = "year_occurrences"


# other columns -> used internally by this package
NODE_NAMES_COLUMN = "node_names"
RESOURCE_COLUMN = "resource"
DOMAIN_PIPE_SUBJECT_ID_COLUMN = "domain_pipe_subject_id_raw"
DOMAIN_PIPE_OBJECT_ID_COLUMN = "domain_pipe_object_id_raw"
OBJECT_ID_RAW_COLUMN = "object_id_raw"
SUBJECT_ID_RAW_COLUMN = "subject_id_raw"
