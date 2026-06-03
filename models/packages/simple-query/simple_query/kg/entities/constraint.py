from typing import Literal, Self, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator
from query_common.entities.kg.concept import Concept
from query_common.entities.kg.identifier import DomainIdentifier
from query_common.entities.kg.relation import Relation


class TripleConstraintArgument(BaseModel):
    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    value: str
    domain_identifier: DomainIdentifier | None = Field(default=None, validation_alias="domain")
    is_variable: bool = False


class TripleConstraint(BaseModel):
    subject: TripleConstraintArgument
    relation: TripleConstraintArgument
    object: TripleConstraintArgument


class ConstraintFilteringOutputEntry(BaseModel):
    variable: str
    type: Literal["concept", "relation"]
    values: list[Concept] | list[Relation]

    @model_validator(mode="after")
    def check_values_type(self) -> Self:
        if self.type == "concept" and any(not isinstance(v, Concept) for v in self.values):
            raise ValueError("all values must be of type `Concept` when `type` is `concept`")
        if self.type == "relation" and any(not isinstance(v, Relation) for v in self.values):
            raise ValueError("all values must be of type `Relation` when `type` is `relation`")
        return self


ConstraintFilteringOutput: TypeAlias = list[ConstraintFilteringOutputEntry]
