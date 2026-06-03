from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Sequence

    from contextualization.core.contracts.reference_database import ReferenceDatabase
    from contextualization.core.entities.link import PublicationConceptLink
    from contextualization.core.entities.publication import PublicationId

PublicationScorerToken = "publication_scorer"

T = TypeVar("T")


class PublicationScorer(ABC, Generic[T]):
    def __init__(self, config: T, reference_database: "ReferenceDatabase") -> None:
        self.config = config
        self.reference_database = reference_database

    @abstractmethod
    def score(
        self, publication_concept_links: "Sequence[PublicationConceptLink]"
    ) -> dict["PublicationId", float]:
        raise NotImplementedError
