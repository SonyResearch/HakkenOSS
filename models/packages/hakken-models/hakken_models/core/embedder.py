"""Embedder configuration and factory for langchain-compatible text embeddings."""

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel

from hakken_models.core.constants import LLMProvider


class EmbedderConfig(BaseModel):
    """Configuration for a text embedder (langchain-compatible).

    The embedder must expose an ``embed_documents(texts: list[str])``
    method returning ``list[list[float]]`` (compatible with
    ``langchain_core.embeddings.Embeddings``).

    For OpenRouter, use ``provider="openai"`` with
    ``base_url="https://openrouter.ai/api/v1"``.
    """

    provider: LLMProvider = LLMProvider.HUGGINGFACE
    model_name: str
    embedding_dim: int
    trainable: bool = False
    encode_batch_size: int = 512
    base_url: str | None = None
    api_key: str | None = None


def create_embedder(cfg: EmbedderConfig) -> Embeddings:
    """Build a langchain Embeddings instance from config.

    Returns an object with ``embed_documents(texts: list[str]) -> list[list[float]]``.
    """
    provider = cfg.provider

    if provider == LLMProvider.HUGGINGFACE:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as exc:
            raise ImportError("pip install langchain-huggingface") from exc
        return HuggingFaceEmbeddings(model_name=cfg.model_name)

    if provider == LLMProvider.OPENAI:
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as exc:
            raise ImportError("pip install langchain-openai") from exc
        kwargs: dict[str, str] = {"model": cfg.model_name}
        if cfg.base_url:
            kwargs["base_url"] = cfg.base_url
        if cfg.api_key:
            kwargs["api_key"] = cfg.api_key
        return OpenAIEmbeddings(**kwargs)

    if provider == LLMProvider.OLLAMA:
        try:
            from langchain_ollama import OllamaEmbeddings
        except ImportError as exc:
            raise ImportError("pip install langchain-ollama") from exc
        kwargs_ollama: dict[str, str] = {"model": cfg.model_name}
        if cfg.base_url:
            kwargs_ollama["base_url"] = cfg.base_url
        return OllamaEmbeddings(**kwargs_ollama)

    raise ValueError(f"Unknown embedder provider: {provider!r}")
