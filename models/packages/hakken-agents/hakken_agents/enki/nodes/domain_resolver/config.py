from pydantic import Field, model_validator

from hakken_agents.tools.element_resolver.config import ElementResolverConfig


class DomainResolverConfig(ElementResolverConfig):
    """Element resolver config for domain resolution."""

    threshold: float = Field(
        default=0.80,
        description="Similarity threshold above which a candidate domain is considered a match",
    )
    fallback_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "When resolving with allowed_domains, minimum similarity to reassign "
            "an extracted domain to the closest allowed one; below this the entity is dropped."
        ),
    )

    @model_validator(mode="after")
    def _fix_domain_resolver_fields(self) -> "DomainResolverConfig":
        if self.context_fields:
            raise ValueError("context_fields must be empty for domain resolver")
        if self.content_fields != ["name"]:
            raise ValueError(
                f"content_fields must be ['name'] for domain resolver, got {self.content_fields!r}"
            )
        return self
