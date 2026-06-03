from pydantic import Field, model_validator

from hakken_agents.tools.element_resolver.config import ElementResolverConfig


class EntityResolverConfig(ElementResolverConfig):
    """Element resolver config for entity resolution.

    Entities are embedded using their name and description for rich
    similarity matching, and are always scoped to a resolved domain
    via the ``domain_id`` metadata filter.
    """

    threshold: float = Field(
        default=0.85,
        description="Similarity threshold above which a candidate entity is considered a match",
    )

    @model_validator(mode="after")
    def _validate_entity_resolver_fields(self) -> "EntityResolverConfig":
        required = {"name", "description"}
        if set(self.content_fields) != required:
            raise ValueError(
                f"content_fields must be {sorted(required)} for entity resolver, "
                f"got {self.content_fields!r}"
            )
        return self
