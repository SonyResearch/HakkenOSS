import numpy as np
import pytest

from contextualization.core.entities.config.publication_scorer import (
    RecencyPublicationScorerConfig,
)
from contextualization.core.entities.config.reference_database import (
    NdjsonReferenceDatabaseConfig,
)
from contextualization.impl.publication_scorer.recency import RecencyPublicationScorer
from contextualization.impl.reference_database import NdjsonReferenceDatabase


@pytest.fixture
def reference_database(ndjson_publications_path, ndjson_publication_concept_links_path):
    return NdjsonReferenceDatabase(
        NdjsonReferenceDatabaseConfig(
            publications_path=ndjson_publications_path,
            publication_concept_links_path=ndjson_publication_concept_links_path,
        )
    )


class TestRecencyPublicationScorer:
    @pytest.mark.parametrize("use_rank", [True, False])
    def test_score(self, use_rank, reference_database):
        config = RecencyPublicationScorerConfig(use_rank=use_rank)
        scorer = RecencyPublicationScorer(config=config, reference_database=reference_database)

        links = reference_database.get_publication_concept_links_from_concept_ids(
            concept_ids=["concept_id1", "concept_id2", "concept_id3", "concept_id4"], flatten=True
        )
        score_dict = scorer.score(publication_concept_links=links)

        assert len(score_dict) == 3

        if use_rank:
            target_score_dict = {"id1": 0.0, "id2": 0.5, "id3": 1.0}
        else:
            target_score_dict = {"id1": 0.0, "id2": 7 / 11, "id3": 1.0}

        for pub_id in score_dict:
            assert np.isclose(score_dict[pub_id], target_score_dict[pub_id])
