from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from contextualization.core.contracts.publication_vector_database import (
        PublicationVectorDatabase,
    )
    from contextualization.core.contracts.reference_reader import ReferenceReader

PublicationEncoderToken = "publication_encoder"

T = TypeVar("T")


class PublicationEncoder(ABC, Generic[T]):
    def __init__(self, config: T) -> None:
        self.config = config

    @abstractmethod
    def encode_and_store_to_db(
        self,
        reference_reader: "ReferenceReader",
        publication_vector_database: "PublicationVectorDatabase",
        skip_existing: bool = True,
    ):
        raise NotImplementedError
