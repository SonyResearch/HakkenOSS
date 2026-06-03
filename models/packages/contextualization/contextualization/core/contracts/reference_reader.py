from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Iterator

    from contextualization.core.entities.link import PublicationConceptLink
    from contextualization.core.entities.publication import Publication

ReferenceReaderToken = "reference_reader"

T = TypeVar("T")


class ReferenceReader(ABC, Generic[T]):
    def __init__(self, config: T) -> None:
        self.config = config

    @abstractmethod
    def iter_publications(self, num_skips: int = 0) -> "Iterator[Publication]":
        raise NotImplementedError

    @abstractmethod
    def iter_publication_concept_links(
        self, num_skips: int = 0
    ) -> "Iterator[PublicationConceptLink]":
        raise NotImplementedError
