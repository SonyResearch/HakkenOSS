from typing import Any

from langchain_core.prompts import PromptTemplate
from loguru import logger

from hakken_agents.config import PromptConfig


def load_prompt_template_from_config(config: PromptConfig, **kwargs: Any) -> PromptTemplate:
    """Load and render a prompt template from a configuration."""
    logger.info(f"Loading prompt template from {config.prompt_template_path}")
    template = PromptTemplate.from_file(
        template_file=config.prompt_template_path,
        template_format=config.template_format,
        encoding=config.encoding,
    )

    if kwargs:
        logger.info(f"Partially rendering prompt template with {kwargs}")
        template = template.partial(**kwargs)

    return template
