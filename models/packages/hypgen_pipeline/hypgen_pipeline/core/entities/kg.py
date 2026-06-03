from dataclasses import dataclass

from hypgen_pipeline.core.values.defaults import (
    DATE_COLUMN_DEFAULT,
    OCID_OBJECT_COLUMN_DEFAULT,
    OCID_SUBJECT_COLUMN_DEFAULT,
    PAPER_LINK_COLUMN_DEFAULT,
)


@dataclass
class KgEdgesColumns:
    date_column: str = DATE_COLUMN_DEFAULT
    paper_link_column: str = PAPER_LINK_COLUMN_DEFAULT
    ocid_subject_column: str = OCID_SUBJECT_COLUMN_DEFAULT
    ocid_object_column: str = OCID_OBJECT_COLUMN_DEFAULT
