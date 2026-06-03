"""Schemas and utilities for text-based fact representation.

Entity format matches the element resolver content template:
  {{ name }}{% if name_id_raw %} | {{ name_id_raw }}{% endif %} || {{ domain }}
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class EntityText(BaseModel):
    """Entity represented as (name, name_id_raw?, domain)."""

    name: str = ""
    name_id_raw: str | None = None
    domain: str = ""

    def to_content(self) -> str:
        """Build content string for lookup/embedding.

        Matches element resolver template:
          {{ name }}{% if name_id_raw %} | {{ name_id_raw }}{% endif %} || {{ domain }}
        """
        parts = [self.name]
        if self.name_id_raw:
            parts.append(f" | {self.name_id_raw}")
        parts.append(f" || {self.domain}")
        return "".join(parts).strip()


class RelationText(BaseModel):
    """Relation represented by name only."""

    name: str = ""

    def to_content(self) -> str:
        """Build content string for lookup/embedding."""
        return self.name


# (subject, relation, object, timestamp). Timestamp is None when not provided.
TripleText = tuple[EntityText, RelationText, EntityText, float | None]


def parse_triple(fact: TripleText | list[Any]) -> TripleText:
    """Parse a fact from a TripleText tuple or a list [s, r, o] or [s, r, o, t].

    List elements can be EntityText, RelationText, str, or dict.
    """
    if isinstance(fact, tuple) and len(fact) == 4:
        s, r, o, t = fact
        return (
            s if isinstance(s, EntityText) else EntityText.model_validate(s),
            r if isinstance(r, RelationText) else RelationText.model_validate(r),
            o if isinstance(o, EntityText) else EntityText.model_validate(o),
            t,
        )
    if isinstance(fact, (list, tuple)) and len(fact) >= 3:
        s = fact[0] if isinstance(fact[0], EntityText) else EntityText.model_validate(fact[0])
        r = fact[1] if isinstance(fact[1], RelationText) else RelationText.model_validate(fact[1])
        o = fact[2] if isinstance(fact[2], EntityText) else EntityText.model_validate(fact[2])
        t = float(fact[3]) if len(fact) > 3 else None
        return (s, r, o, t)
    raise ValueError(f"Fact must be TripleText or list of 3+ elements, got {type(fact)}")
