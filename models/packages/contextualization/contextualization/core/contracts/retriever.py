from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Sequence

    from contextualization.core.contracts.context_summarizer import ContextSummarizer
    from contextualization.core.contracts.publication_scorer import PublicationScorer
    from contextualization.core.contracts.reference_database import ReferenceDatabase
    from contextualization.core.entities.retrieval import RetrievalReturnType, RetrievedContext
    from contextualization.core.entities.triple import Triple

RetrieverToken = "retriever"

T = TypeVar("T")


class Retriever(ABC, Generic[T]):
    def __init__(
        self,
        config: T,
        reference_database: "ReferenceDatabase",
        publication_scorer: "PublicationScorer",
        context_summarizer: "ContextSummarizer | None" = None,
    ) -> None:
        self.config = config
        self.reference_database = reference_database
        self.publication_scorer = publication_scorer
        self.context_summarizer = context_summarizer

    @abstractmethod
    def retrieve(
        self,
        triples: "Sequence[Triple]",
        max_num_references: int,
        return_type: "RetrievalReturnType",
    ) -> "RetrievedContext":
        raise NotImplementedError
