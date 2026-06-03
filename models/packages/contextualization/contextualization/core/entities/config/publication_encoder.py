from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from contextualization.core.values.types import PublicationEncoderType


class PublicationEncoderConfigBase(BaseModel):
    config_type: PublicationEncoderType


class LLMPublicationEncoderConfig(PublicationEncoderConfigBase):
    config_type: Literal[PublicationEncoderType.LLM] = PublicationEncoderType.LLM

    hf_model_name_or_path: str = Field(
        default="Qwen/Qwen3-Embedding-0.6B",
        description=(
            "Model name or path to be passed to `transformers.AutoModel.from_pretrained(...)`."
        ),
    )
    max_length: int | None = Field(
        default=8192,
        description=(
            "Size (number of tokens) of text chunks. "
            "If `None`, it will not split a text into chunks."
        ),
    )
    pooling_method: Literal["last", "avg"] = Field(
        default="last",
        description="The pooling method to apply on hidden states from the last layer.",
    )

    title_max_ratio: float = Field(
        default=0.5,
        description=(
            "The maximum ratio of title tokens in the input, if truncation is needed. "
            "Note that if no truncation is required for joining a title and an abstract, "
            "it will be ignored."
        ),
    )
    overlap: int = Field(
        default=128, description="Number of overlapping tokens between neighboring chunks."
    )
    hf_model_loading_kwargs: dict[str, Any] = Field(
        default={},
        description=(
            "Keyword arguments that will be passed to "
            "`transformers.AutoModel.from_pretrained(...)`."
        ),
    )
    tokenizer_kwargs: dict[str, Any] = Field(
        default={"dtype": "float16", "attn_implementation": "flash_attention_2"},
        description=(
            "Keyword arguments that will be additionally passed to "
            "`AutoTokenizer.from_pretrained(...)`. "
        ),
    )
    device: str | int | None = Field(
        default=None,
        description=(
            "Device name or ID to which a model will be moved. "
            "Should never be set in a distributed environment."
        ),
    )
    batch_size: int = Field(default=32, description="Size of batches for encoding.")

    title_prefix: str = Field(
        default="Title: ", description="Prefix to be prepended to title text."
    )
    abstract_prefix: str = Field(
        default="Abstract: ", description="Prefix to be prepended to abstract text."
    )
    joiner: str = Field(default="\n", description="Joiner used when joining title and abstract.")


PublicationEncoderConfig = Annotated[
    LLMPublicationEncoderConfig, Field(discriminator="config_type")
]
