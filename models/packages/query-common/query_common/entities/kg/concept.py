from __future__ import annotations

from pydantic import BaseModel

from query_common.entities.kg.identifier import ConceptIdentifier, DomainIdentifier


class Concept(BaseModel):
    identifier: ConceptIdentifier
    label: str = ""
    domain_identifier: DomainIdentifier | None = None

    def model_post_init(self, __context) -> None:
        if self.label == "":
            self.label = f"concept_{self.identifier}"
