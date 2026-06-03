from __future__ import annotations

from pydantic import BaseModel

from query_common.entities.kg.identifier import ConceptIdentifier, RelationIdentifier


class Triple(BaseModel):
    subject_identifier: ConceptIdentifier
    relation_identifier: RelationIdentifier
    object_identifier: ConceptIdentifier
