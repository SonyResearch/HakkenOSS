from pydantic import Field
from pydantic_settings import BaseSettings


class PromptConfig(BaseSettings):
    prompt_template_path: str = Field(
        default="hakken_agents/graph_builder/prompts/v0.yaml",
        description="The path to the prompt template to use for entity extraction",
    )
    template_format: str = Field(
        default="jinja2",
        description="The format of the prompt template",
    )
    encoding: str = Field(
        default="utf-8",
        description="The encoding of the prompt template",
    )
