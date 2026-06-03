from pydantic import BaseModel, Field

from hakken_agents.config import LLMConfig


class InfoExtractorConfig(BaseModel):
    system_prompt_id: str = Field(description="Prompt-registry ID for the system prompt")
    user_prompt_id: str = Field(description="Prompt-registry ID for the user (Jinja2) prompt")
    llm: LLMConfig = Field(description="The LLM to use for info extraction")
