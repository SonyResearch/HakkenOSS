import numpy as np
import pytest

from contextualization.core.entities.config.publication_scorer import (
    CoveragePublicationScorerConfig,
)
from contextualization.core.entities.config.reference_database import (
    NdjsonReferenceDatabaseConfig,
)
from contextualization.impl.publication_scorer.coverage import CoveragePublicationScorer
from contextualization.impl.reference_database import NdjsonReferenceDatabase


@pytest.fixture
def reference_database(ndjson_publications_path, ndjson_publication_concept_links_path):
    return NdjsonReferenceDatabase(
        NdjsonReferenceDatabaseConfig(
            publications_path=ndjson_publications_path,
            publication_concept_links_path=ndjson_publication_concept_links_path,
        )
    )


class TestCoveragePublicationScorer:
    def test_numbers_of_concepts_for_publications(self, reference_database):
        config = CoveragePublicationScorerConfig()
        scorer = CoveragePublicationScorer(config=config, reference_database=reference_database)
        scores = scorer._numbers_of_concepts_for_publications(publication_ids=["id1", "id2", "id3"])
        assert np.allclose([scores["id1"], scores["id2"], scores["id3"]], [2, 1, 1])

    def test_score(self, reference_database):
        config = CoveragePublicationScorerConfig()
        scorer = CoveragePublicationScorer(config=config, reference_database=reference_database)

        links = reference_database.get_publication_concept_links_from_concept_ids(
            concept_ids=["concept_id2", "concept_id3"], flatten=True
        )
        score_dict = scorer.score(publication_concept_links=links)

        assert len(score_dict) == 3
        target_score_dict = {"id1": 1 / 2, "id2": 1, "id3": 1}

        for pub_id in score_dict:
            assert np.isclose(score_dict[pub_id], target_score_dict[pub_id])
