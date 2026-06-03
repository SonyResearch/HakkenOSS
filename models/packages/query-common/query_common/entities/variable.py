from __future__ import annotations

from pydantic import BaseModel

from query_common.entities.kg.identifier import DomainIdentifier

VarLabel = str


class Variable(BaseModel):
    """e.g. x in Drugs"""

    label: VarLabel
    domain_identifier: DomainIdentifier
