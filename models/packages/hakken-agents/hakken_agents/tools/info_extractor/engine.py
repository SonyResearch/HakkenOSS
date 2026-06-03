from typing import Any, Generic, Self, TypeVar, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from loguru import logger
from pydantic import BaseModel

from hakken_agents.config import LLMConfig
from hakken_agents.utils.llm import get_llm

from .config import InfoExtractorConfig

T = TypeVar("T", bound=BaseModel)
ConfigT = TypeVar("ConfigT", bound=InfoExtractorConfig)


class InfoExtractor(Generic[T, ConfigT]):
    """Base class for info extractors. Subclasses must define output_schema class attribute."""

    output_schema: type[T]  # Subclasses must define this

    def __init__(
        self,
        system_prompt: str,
        user_prompt: PromptTemplate,
        llm_config: LLMConfig,
        output_schema: type[T],
    ) -> None:
        self.system_msg = SystemMessage(content=system_prompt)
        self.user_prompt = user_prompt

        self.llm = (
            get_llm(llm_config)
            .with_structured_output(output_schema)
            .with_retry(
                stop_after_attempt=3,
                wait_exponential_jitter=True,
            )
        )

    @property
    def allowed_user_variables(self) -> list[str]:
        return []

    @property
    def user_variables_are_required(self) -> bool:
        return False

    def verify_user_variables(self, user_variables: dict[str, Any] | None = None) -> None:
        if user_variables is None and self.user_variables_are_required:
            raise ValueError("user variables are required")
        if user_variables is not None:
            for variable in user_variables:
                if variable not in self.allowed_user_variables:
                    raise ValueError(f"invalid user variable: {variable}")

    async def arun(self, text: str, user_variables: dict[str, Any] | None = None) -> T:
        self.verify_user_variables(user_variables)
        variables = {}
        variables["content"] = text
        if user_variables is not None:
            variables.update(user_variables)

        user_message = HumanMessage(content=self.user_prompt.format(**variables))

        messages = [self.system_msg, user_message]
        logger.info("Starting info extraction")
        response = await self.llm.ainvoke(messages)
        logger.info("Info extraction finished")
        return cast(T, response)

    def run(self, text: str, user_variables: dict[str, Any] | None = None) -> T:
        self.verify_user_variables(user_variables)
        variables = {}
        variables["content"] = text
        if user_variables is not None:
            variables.update(user_variables)

        user_message = HumanMessage(content=self.user_prompt.format(**variables))

        messages = [self.system_msg, user_message]
        logger.info("Starting info extraction")
        response = self.llm.invoke(messages)
        logger.info("Info extraction finished")
        return cast(T, response)

    @classmethod
    def from_config(cls, config: ConfigT) -> Self:
        output_schema = getattr(cls, "output_schema", None)
        if output_schema is None:
            raise TypeError(
                f"{cls.__name__} must define 'output_schema' class attribute "
                "(the Pydantic model for structured LLM output)"
            )

        from hakken_agents.enki.prompts.registry import prompt_registry

        system_prompt = prompt_registry.get(config.system_prompt_id)
        user_template = prompt_registry.get(config.user_prompt_id)
        user_prompt = PromptTemplate(
            template=user_template,
            template_format="jinja2",
        )

        return cls(system_prompt, user_prompt, config.llm, output_schema)
