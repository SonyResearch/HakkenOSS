from __future__ import annotations

from pydantic import BaseModel

from query_common.entities.kg.identifier import RelationIdentifier


class Relation(BaseModel):
    identifier: RelationIdentifier
    label: str = ""

    def model_post_init(self, __context) -> None:
        if self.label == "":
            self.label = self.identifier
