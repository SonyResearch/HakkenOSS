from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Sequence

    from contextualization.core.entities.retrieval import Reference

ContextSummarizerToken = "context_summarizer"

T = TypeVar("T")


class ContextSummarizer(ABC, Generic[T]):
    def __init__(self, config: T) -> None:
        self.config = config

    @abstractmethod
    def summarize_all_references(self, references: "Sequence[Reference]") -> str:
        raise NotImplementedError

    @abstractmethod
    def summarize_reference(self, reference: "Reference") -> str:
        raise NotImplementedError
