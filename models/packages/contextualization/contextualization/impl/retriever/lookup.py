import warnings
from typing import TYPE_CHECKING

from contextualization.core.contracts.retriever import Retriever
from contextualization.core.entities.config.retriever import LookupRetrieverConfig
from contextualization.core.entities.retrieval import (
    Reference,
    RetrievalReturnType,
    RetrievedContext,
)
from contextualization.core.values.errors import RetrievalWarning

if TYPE_CHECKING:
    from collections.abc import Sequence

    from contextualization.core.entities.link import PublicationConceptLink
    from contextualization.core.entities.triple import Triple


class LookupRetriever(Retriever[LookupRetrieverConfig]):
    def retrieve(
        self, triples: "Sequence[Triple]", max_num_references: int, return_type: RetrievalReturnType
    ) -> RetrievedContext:
        if return_type == RetrievalReturnType.TEXT:
            # TODO: Remove this
            warnings.warn(
                f"Return type {return_type} is not supported in lookup retriever and "
                "`text` attribute will have `None` value. "
                "Consider using `title` and `abstract` of publication instead.",
                RetrievalWarning,
                stacklevel=2,
            )
        if return_type == RetrievalReturnType.SUMMARY and self.context_summarizer is None:
            warnings.warn(
                f"Got return type {return_type} but context summarizer is not set. "
                "`summary` attribute will have `None` value.",
                RetrievalWarning,
                stacklevel=2,
            )
        subjects = [triple.subject for triple in triples]
        objects = [triple.object for triple in triples]
        concept_ids = list(set(subjects + objects))

        links: list[PublicationConceptLink] = (
            self.reference_database.get_publication_concept_links_from_concept_ids(
                concept_ids, flatten=True
            )
        )
        publication_score_dict = self.publication_scorer.score(links)

        publication_ids_sorted = [
            pub_id
            for pub_id, _ in sorted(
                publication_score_dict.items(), key=lambda item: item[1], reverse=True
            )
        ]
        publication_ids_sorted = publication_ids_sorted[:max_num_references]
        reference_publications = self.reference_database.get_publications(publication_ids_sorted)

        references: list[Reference] = []
        for publication in reference_publications:
            references.append(
                Reference(
                    publication_info=publication,
                    score=publication_score_dict[publication.publication_id],
                    text=None,
                )
            )

        summary: str | None = None
        if return_type == RetrievalReturnType.SUMMARY and self.context_summarizer is not None:
            summary = self.context_summarizer.summarize_all_references(references)

            for reference in references:
                reference.summary = self.context_summarizer.summarize_reference(reference)

        return RetrievedContext(references=references, summary=summary)
