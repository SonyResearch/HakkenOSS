from pydantic import Field

from hakken_agents.tools.info_extractor.config import InfoExtractorConfig


class FactExtractorConfig(InfoExtractorConfig):
    """Config for fact extraction from text.

    Extends the base InfoExtractorConfig with soft relation-type guidance fields.
    """

    preferred_relation_types: list[str] = Field(
        default_factory=list,
        description=(
            "Preferred relation type names injected as soft guidance"
            " into the prompt. Empty means no preference."
        ),
    )
    use_relevant_relation_types: bool = Field(
        default=True,
        description=(
            "Whether to retrieve relevant relation types from the vector store for prompt guidance"
        ),
    )
