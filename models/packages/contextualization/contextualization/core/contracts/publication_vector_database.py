from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from contextualization.core.entities.publication import PublicationId
    from contextualization.core.entities.vector_database import (
        Metadata,
        VectorDatabaseStatistics,
        VectorWithMetadata,
    )

PublicationVectorDatabaseToken = "publication_vector_database"

T = TypeVar("T")


class PublicationVectorDatabase(ABC, Generic[T]):
    def __init__(self, config: T) -> None:
        self.config = config

    @abstractmethod
    def exists(self, publication_id: "PublicationId", chunk_index: int | None = None) -> bool:
        raise NotImplementedError

    @abstractmethod
    def retrieve_statistics(self) -> "VectorDatabaseStatistics":
        raise NotImplementedError

    @abstractmethod
    def metadata_iterator(self, include_text: bool = False) -> "Iterator[Metadata]":
        raise NotImplementedError

    @abstractmethod
    def get_by_publication_ids(
        self,
        publication_ids: "Sequence[PublicationId]",
        include_text: bool = False,
    ) -> "Sequence[VectorWithMetadata]":
        raise NotImplementedError

    @abstractmethod
    def insert(self, data: "Sequence[VectorWithMetadata]") -> None:
        raise NotImplementedError
