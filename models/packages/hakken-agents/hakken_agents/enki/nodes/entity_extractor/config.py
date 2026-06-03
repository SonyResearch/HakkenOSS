from pydantic import Field

from hakken_agents.tools.info_extractor.config import InfoExtractorConfig


class EntityExtractorConfig(InfoExtractorConfig):
    allowed_domains: list[str] | None = Field(
        default=None, description="Allowed domains for entity extraction"
    )
    use_relevant_domains: bool = Field(
        default=True,
        description="Whether to use relevant domains for entity extraction",
    )
