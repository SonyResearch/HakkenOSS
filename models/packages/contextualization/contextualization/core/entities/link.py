from typing import TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from contextualization.core.entities.publication import PublicationId
from contextualization.core.entities.types import StripString

ConceptId: TypeAlias = StripString


class PublicationConceptLink(BaseModel):
    # To automatically convert integer concept IDs to strings
    model_config = ConfigDict(
        coerce_numbers_to_str=True, validate_by_name=True, validate_by_alias=True
    )

    publication_id: PublicationId = Field(validation_alias="pmid")
    concept_id: ConceptId = Field(validation_alias="node_id")
