from typing import Annotated, TypeAlias

from pydantic import BeforeValidator

ConceptIdentifier: TypeAlias = Annotated[str, BeforeValidator(str.strip)]
RelationIdentifier: TypeAlias = Annotated[str, BeforeValidator(str.strip)]
DomainIdentifier: TypeAlias = Annotated[str, BeforeValidator(str.strip)]
