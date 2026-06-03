import itertools
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, Literal, TypeVar, overload

if TYPE_CHECKING:
    from collections.abc import Sequence

    from contextualization.core.entities.link import ConceptId, PublicationConceptLink
    from contextualization.core.entities.publication import Publication, PublicationId

ReferenceDatabaseToken = "reference_database"

T = TypeVar("T")


class ReferenceDatabase(ABC, Generic[T]):
    def __init__(self, config: T) -> None:
        self.config = config

    def get_publication(self, publication_id: "PublicationId") -> "Publication":
        return self.get_publications([publication_id])[0]

    @abstractmethod
    def get_publications(self, publication_ids: "Sequence[PublicationId]") -> list["Publication"]:
        """
        Get a list of `Publication`s given a list of publication_encoder IDs.
        Resulting list has the same order with the input list.
        If a publication_encoder is not found for any of the given publication_encoder IDs,
        it will raise a `PublicationNotFoundException`.
        """
        raise NotImplementedError

    @abstractmethod
    def _get_publication_concept_links_from_publication_ids(
        self,
        publication_ids: "Sequence[PublicationId]",
        per_publication_max_size: int | None = None,
    ) -> list[list["PublicationConceptLink"]]:
        raise NotImplementedError

    @abstractmethod
    def _get_publication_concept_links_from_concept_ids(
        self,
        concept_ids: "Sequence[ConceptId]",
        per_concept_max_size: int | None = None,
    ) -> list[list["PublicationConceptLink"]]:
        """
        Get a list of lists of `PublicationConceptLink`s given a list of concept IDs.
        Resulting list has the same order with the input list.
        If there is no publication-concept link for an concept ID in the input list,
        the corresponding element will be an empty list.
        """
        raise NotImplementedError

    @overload
    def get_publication_concept_links_from_publication_ids(
        self,
        publication_ids: "Sequence[PublicationId]",
        flatten: Literal[True] = True,
        per_publication_max_size: int | None = None,
    ) -> list["PublicationConceptLink"]: ...

    @overload
    def get_publication_concept_links_from_publication_ids(
        self,
        publication_ids: "Sequence[PublicationId]",
        flatten: Literal[False],
        per_publication_max_size: int | None = None,
    ) -> list[list["PublicationConceptLink"]]: ...

    def get_publication_concept_links_from_publication_ids(
        self,
        publication_ids: "Sequence[PublicationId]",
        flatten: bool = True,
        per_publication_max_size: int | None = None,
    ) -> list["PublicationConceptLink"] | list[list["PublicationConceptLink"]]:
        links_list = self._get_publication_concept_links_from_publication_ids(
            publication_ids, per_publication_max_size=per_publication_max_size
        )
        if flatten:
            return list(itertools.chain.from_iterable(links_list))
        return links_list

    @overload
    def get_publication_concept_links_from_concept_ids(
        self,
        concept_ids: "Sequence[ConceptId]",
        flatten: Literal[True] = True,
        per_concept_max_size: int | None = None,
    ) -> list["PublicationConceptLink"]: ...

    @overload
    def get_publication_concept_links_from_concept_ids(
        self,
        concept_ids: "Sequence[ConceptId]",
        flatten: Literal[False],
        per_concept_max_size: int | None = None,
    ) -> list[list["PublicationConceptLink"]]: ...

    def get_publication_concept_links_from_concept_ids(
        self,
        concept_ids: "Sequence[ConceptId]",
        flatten: bool = True,
        per_concept_max_size: int | None = None,
    ) -> list["PublicationConceptLink"] | list[list["PublicationConceptLink"]]:
        links_list = self._get_publication_concept_links_from_concept_ids(
            concept_ids, per_concept_max_size=per_concept_max_size
        )
        if flatten:
            return list(itertools.chain.from_iterable(links_list))
        return links_list
