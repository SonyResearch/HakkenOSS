from typing import Annotated, Literal

from pydantic import BaseModel, Field

from contextualization.core.values.types import ContextSummarizerType


class ContextSummarizerConfigBase(BaseModel):
    config_type: ContextSummarizerType


class LLMContextSummarizerConfig(ContextSummarizerConfigBase):
    config_type: Literal[ContextSummarizerType.LLM] = ContextSummarizerType.LLM
    hf_model_name_or_path: str = "Qwen/Qwen3-4B-Instruct-2507"
    prompt_all_references: str = (
        "Summarize the papers that will given in 1 paragraph, "
        "using values in <title> and <abstract>. "
        "If needed, refer the paper using the value in <id>."
    )
    prompt_single_reference: str = (
        "Summarize the paper that will given in maximum 2 sentences, "
        "using values in <title> and <abstract>. "
        "If needed, refer the paper using the value in <id>."
    )
    device: str | int = "cpu"
    max_new_tokens: int = 512


ContextSummarizerConfig = Annotated[LLMContextSummarizerConfig, Field(discriminator="config_type")]
