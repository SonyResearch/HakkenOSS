from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from contextualization.core.contracts.context_summarizer import ContextSummarizer
from contextualization.core.entities.config.publication_scorer import (
    CoveragePublicationScorerConfig,
)
from contextualization.core.entities.config.reference_database import (
    NdjsonReferenceDatabaseConfig,
)
from contextualization.core.entities.config.retriever import LookupRetrieverConfig
from contextualization.core.entities.retrieval import RetrievalReturnType
from contextualization.core.entities.triple import Triple
from contextualization.core.values.errors import RetrievalWarning
from contextualization.impl.publication_scorer.coverage import CoveragePublicationScorer
from contextualization.impl.reference_database import NdjsonReferenceDatabase
from contextualization.impl.retriever.lookup import LookupRetriever

if TYPE_CHECKING:
    from contextualization.core.contracts.publication_scorer import PublicationScorer
    from contextualization.core.contracts.reference_database import ReferenceDatabase


@pytest.fixture
def reference_database(
    ndjson_publications_path, ndjson_publication_concept_links_path
) -> "ReferenceDatabase":
    return NdjsonReferenceDatabase(
        config=NdjsonReferenceDatabaseConfig(
            publications_path=ndjson_publications_path,
            publication_concept_links_path=ndjson_publication_concept_links_path,
        ),
    )


@pytest.fixture
def publication_scorer(reference_database) -> "PublicationScorer":
    return CoveragePublicationScorer(
        config=CoveragePublicationScorerConfig(), reference_database=reference_database
    )


class TestLookupRetriever:
    def test_retrieve(self, reference_database, publication_scorer):
        retriever = LookupRetriever(
            config=LookupRetrieverConfig(),
            reference_database=reference_database,
            publication_scorer=publication_scorer,
        )
        triples = [
            Triple(subject="concept_id1", relation="rel1", object="concept_id2"),
            Triple(subject="concept_id2", relation="rel2", object="concept_id1"),
        ]
        retrieved_context = retriever.retrieve(
            triples=triples, max_num_references=100, return_type=RetrievalReturnType.PUBLICATION
        )
        retrieved_context_2 = retriever.retrieve(
            triples=triples, max_num_references=2, return_type=RetrievalReturnType.PUBLICATION
        )
        for ref in retrieved_context_2.references:
            assert ref in retrieved_context.references

    def test_with_summary_return_type(self, reference_database, publication_scorer):
        mock_summarizer = MagicMock(spec=ContextSummarizer)
        mock_summarizer.summarize_all_references.return_value = "summary_1"
        mock_summarizer.summarize_reference.return_value = "summary_2"

        retriever = LookupRetriever(
            config=LookupRetrieverConfig(),
            reference_database=reference_database,
            publication_scorer=publication_scorer,
            context_summarizer=mock_summarizer,
        )
        triples = [
            Triple(subject="concept_id1", relation="rel1", object="concept_id2"),
            Triple(subject="concept_id2", relation="rel2", object="concept_id1"),
        ]
        retrieved_context = retriever.retrieve(
            triples=triples, max_num_references=100, return_type=RetrievalReturnType.SUMMARY
        )
        retrieved_context_2 = retriever.retrieve(
            triples=triples, max_num_references=2, return_type=RetrievalReturnType.SUMMARY
        )

        for ref in retrieved_context_2.references:
            assert ref in retrieved_context.references
            assert ref.summary == "summary_2"
        assert retrieved_context_2.summary == "summary_1"

    def test_retrieve_warning_with_text_return_type(self, reference_database, publication_scorer):
        retriever = LookupRetriever(
            config=LookupRetrieverConfig(),
            reference_database=reference_database,
            publication_scorer=publication_scorer,
        )
        triples = [
            Triple(subject="concept_id1", relation="rel1", object="concept_id2"),
            Triple(subject="concept_id2", relation="rel2", object="concept_id1"),
        ]
        with pytest.warns(RetrievalWarning):
            retrieved_context = retriever.retrieve(
                triples=triples, max_num_references=100, return_type=RetrievalReturnType.TEXT
            )
            retrieved_context_2 = retriever.retrieve(
                triples=triples, max_num_references=2, return_type=RetrievalReturnType.TEXT
            )
        for ref in retrieved_context_2.references:
            assert ref in retrieved_context.references
            assert ref.text is None

    def test_retrieve_warning_with_summary_return_type(
        self, reference_database, publication_scorer
    ):
        retriever = LookupRetriever(
            config=LookupRetrieverConfig(),
            reference_database=reference_database,
            publication_scorer=publication_scorer,
        )
        triples = [
            Triple(subject="concept_id1", relation="rel1", object="concept_id2"),
            Triple(subject="concept_id2", relation="rel2", object="concept_id1"),
        ]
        with pytest.warns(RetrievalWarning):
            retrieved_context = retriever.retrieve(
                triples=triples, max_num_references=100, return_type=RetrievalReturnType.SUMMARY
            )
            retrieved_context_2 = retriever.retrieve(
                triples=triples, max_num_references=2, return_type=RetrievalReturnType.SUMMARY
            )
        for ref in retrieved_context_2.references:
            assert ref in retrieved_context.references
        assert retrieved_context.summary is None
        assert retrieved_context_2.summary is None
