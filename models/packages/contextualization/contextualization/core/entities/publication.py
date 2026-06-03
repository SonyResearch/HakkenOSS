from typing import TypeAlias

from loguru import logger
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from contextualization.core.entities.types import StripString

PublicationId: TypeAlias = StripString


class Author(BaseModel):
    model_config = ConfigDict(coerce_numbers_to_str=True)

    first_name: StripString | None = None
    last_name: StripString | None = None

    @model_validator(mode="after")
    def check_at_least_one_name_is_given(self) -> "Author":
        if not self.first_name and not self.last_name:
            raise ValueError("at least one of `first_name` and `last_name` should be given")
        return self


class Publication(BaseModel):
    model_config = ConfigDict(
        coerce_numbers_to_str=True, validate_by_name=True, validate_by_alias=True
    )

    publication_id: PublicationId = Field(
        description="PubMed ID of the publication.", validation_alias="pmid"
    )
    year: PositiveInt = Field(description="Publication year.")
    title: StripString = Field(description="Publication title.")
    abstract: StripString | None = Field(description="Abstract for the publication")
    doi: StripString | None = Field(default=None, description="DOI of the publication.")
    authors: list[Author] = Field(default=[], description="Author information.")
    citations_count: PositiveInt | None = Field(
        default=None, description="The number of publications citing the publication."
    )

    @field_validator("authors", mode="before")
    @classmethod
    def filter_invalid_authors(cls, v: list, info: ValidationInfo) -> list:
        if not isinstance(v, list):
            return v

        valid_authors = []
        for author_data in v:
            try:
                Author.model_validate(author_data)
                valid_authors.append(author_data)
            except ValidationError:
                pub_id = info.data.get("publication_id")
                logger.warning(
                    f"Validation error for processing the author data `{author_data!s}` in `{v!s}` "
                    f"while processing {pub_id}; ignore the author."
                )

        return valid_authors
