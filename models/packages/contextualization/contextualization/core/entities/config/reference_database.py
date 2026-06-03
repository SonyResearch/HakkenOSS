from typing import Annotated, Literal

from pydantic import BaseModel, Field, FilePath

from contextualization.core.values.types import ReferenceDatabaseType


class ReferenceDatabaseConfigBase(BaseModel):
    config_type: ReferenceDatabaseType


class NdjsonReferenceDatabaseConfig(ReferenceDatabaseConfigBase):
    config_type: Literal[ReferenceDatabaseType.NDJSON] = ReferenceDatabaseType.NDJSON

    publications_path: FilePath
    publication_concept_links_path: FilePath


class PostgresReferenceDatabaseConfig(ReferenceDatabaseConfigBase):
    config_type: Literal[ReferenceDatabaseType.POSTGRES] = ReferenceDatabaseType.POSTGRES
    connection_string: str = Field(
        description=(
            "Connection string in a format `postgresql://...` or "
            "a whitespace-separated `key=value` pairs"
        )
    )
    publication_table_name: str = "publication"
    publication_concept_link_table_name: str = "publication_concept"


ReferenceDatabaseConfig = Annotated[
    NdjsonReferenceDatabaseConfig | PostgresReferenceDatabaseConfig,
    Field(discriminator="config_type"),
]
