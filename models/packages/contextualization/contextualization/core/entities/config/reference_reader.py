from typing import Annotated, Literal, Self

from pydantic import BaseModel, DirectoryPath, Field, model_validator

from contextualization.core.values.types import ReferenceReaderType


class ReferenceReaderConfigBase(BaseModel):
    config_type: ReferenceReaderType


class ParquetReferenceReaderConfig(ReferenceReaderConfigBase):
    config_type: Literal[ReferenceReaderType.PARQUET] = ReferenceReaderType.PARQUET

    publications_directory: DirectoryPath | None = None
    publication_concept_links_directory: DirectoryPath | None = None

    @model_validator(mode="after")
    def check_directory_paths(self) -> Self:
        if not self.publications_directory and not self.publication_concept_links_directory:
            raise ValueError(
                "at least one of `publications_directory` or `publication_concept_links_directory` "
                "should be given"
            )
        return self


class NdjsonReferenceReaderConfig(ReferenceReaderConfigBase):
    config_type: Literal[ReferenceReaderType.NDJSON] = ReferenceReaderType.NDJSON

    publications_directory: DirectoryPath | None = None
    publication_concept_links_directory: DirectoryPath | None = None

    @model_validator(mode="after")
    def check_directory_paths(self) -> Self:
        if not self.publications_directory and not self.publication_concept_links_directory:
            raise ValueError(
                "at least one of `publications_directory` or `publication_concept_links_directory` "
                "should be given"
            )
        return self


ReferenceReaderConfig = Annotated[
    ParquetReferenceReaderConfig | NdjsonReferenceReaderConfig, Field(discriminator="config_type")
]
