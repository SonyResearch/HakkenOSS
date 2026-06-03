from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
import pytest

from contextualization.core.entities.config.publication_scorer import (
    AggregatedPublicationScorerConfig,
    CoveragePublicationScorerConfig,
    RecencyPublicationScorerConfig,
)
from contextualization.core.entities.config.reference_database import (
    NdjsonReferenceDatabaseConfig,
)
from contextualization.impl.publication_scorer.aggregated import (
    AggregatedPublicationScorer,
)
from contextualization.impl.reference_database import NdjsonReferenceDatabase

if TYPE_CHECKING:
    from contextualization.core.entities.publication import PublicationId


@pytest.fixture
def reference_database(ndjson_publications_path, ndjson_publication_concept_links_path):
    return NdjsonReferenceDatabase(
        NdjsonReferenceDatabaseConfig(
            publications_path=ndjson_publications_path,
            publication_concept_links_path=ndjson_publication_concept_links_path,
        )
    )


@pytest.fixture
def coverage_publication_scorer_config():
    return CoveragePublicationScorerConfig()


@pytest.fixture
def recency_publication_scorer_config():
    return RecencyPublicationScorerConfig(use_rank=True)


@pytest.fixture
def coverage_target_score_dict():
    return {
        "id1": 1 / 2,
        "id2": 1,
        "id3": 1,
    }


@pytest.fixture
def recency_target_score_dict():
    return {
        "id1": 0.0,
        "id2": 0.5,
        "id3": 1.0,
    }


@pytest.fixture
def publication_scorer_configs(
    coverage_publication_scorer_config, recency_publication_scorer_config
):
    return [coverage_publication_scorer_config, recency_publication_scorer_config]


@pytest.fixture
def target_score_dicts(coverage_target_score_dict, recency_target_score_dict):
    return [coverage_target_score_dict, recency_target_score_dict]


class TestCoveragePublicationScorer:
    def test_score(self, publication_scorer_configs, target_score_dicts, reference_database):
        config = AggregatedPublicationScorerConfig(
            aggregation_configs=[
                {"weight": 0.8, "config": config} for config in publication_scorer_configs
            ]
        )
        scorer = AggregatedPublicationScorer(config=config, reference_database=reference_database)

        links = reference_database.get_publication_concept_links_from_concept_ids(
            concept_ids=["concept_id2", "concept_id3"], flatten=True
        )
        score_dict = scorer.score(publication_concept_links=links)

        aggregated_target_score_dict: dict[PublicationId, float] = defaultdict(float)
        for target_score_dict in target_score_dicts:
            for publication_id, score in target_score_dict.items():
                aggregated_target_score_dict[publication_id] += 0.8 * score

        for pub_id in score_dict:
            assert np.isclose(score_dict[pub_id], aggregated_target_score_dict[pub_id])
