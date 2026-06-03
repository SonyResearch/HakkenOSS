from langchain.chat_models.base import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from hakken_agents.config.llm import LLMConfig


def get_llm(config: LLMConfig) -> BaseChatModel:
    if config.is_ollama():
        ollama_kwargs: dict = dict(
            model=config.name,
            reasoning=None,
            temperature=config.temperature,
            base_url=config.base_url,
        )
        if config.max_tokens is not None:
            ollama_kwargs["num_predict"] = config.max_tokens
        return ChatOllama(**ollama_kwargs)
    kwargs: dict = dict(
        model=config.name,
        temperature=config.temperature,
        api_key=config.api_key.get_secret_value() if config.api_key else None,
        base_url=config.base_url,
    )
    if config.max_tokens is not None:
        kwargs["max_tokens"] = config.max_tokens
    return ChatOpenAI(**kwargs)
