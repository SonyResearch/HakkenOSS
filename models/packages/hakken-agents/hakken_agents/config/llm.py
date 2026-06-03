from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        default="xiaomi/mimo-v2-flash:free",
        description="The name of the LLM model to use for entity extraction",
    )
    temperature: float = Field(
        default=0.0,
        description="Temperature setting for the LLM (0.0 for deterministic output)",
    )
    api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(""),
        description="API key for authenticating with the LLM service",
    )

    @field_validator("api_key", mode="before")
    @classmethod
    def _coerce_api_key(cls, v: object) -> SecretStr:
        if v is None or v == "":
            return SecretStr("")
        if isinstance(v, str):
            return SecretStr(v)
        return v

    max_tokens: int | None = Field(
        default=None,
        description="Maximum number of tokens the LLM may generate. None means no limit.",
    )

    base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="Base URL for the LLM API endpoint",
    )

    def is_ollama(self) -> bool:
        return len(self.api_key.get_secret_value()) == 0
