from typing import TYPE_CHECKING

from contextualization.core.contracts.publication_scorer import PublicationScorer
from contextualization.core.entities.config.publication_scorer import (
    AggregatedPublicationScorerConfig,
    CoveragePublicationScorerConfig,
    RecencyPublicationScorerConfig,
)
from contextualization.core.values.errors import ConfigurationError
from contextualization.impl.publication_scorer import CoveragePublicationScorer
from contextualization.impl.publication_scorer.recency import RecencyPublicationScorer

if TYPE_CHECKING:
    from collections.abc import Sequence

    from contextualization.core.contracts.reference_database import ReferenceDatabase
    from contextualization.core.entities.link import PublicationConceptLink
    from contextualization.core.entities.publication import PublicationId


class AggregatedPublicationScorer(PublicationScorer[AggregatedPublicationScorerConfig]):
    def __init__(
        self, config: AggregatedPublicationScorerConfig, reference_database: "ReferenceDatabase"
    ) -> None:
        super().__init__(config=config, reference_database=reference_database)

        self.weights: list[float] = []
        self.scorers: list[PublicationScorer] = []
        for aggregation_config in config.aggregation_configs:
            self.weights.append(aggregation_config.weight)
            if isinstance(aggregation_config.config, CoveragePublicationScorerConfig):
                self.scorers.append(
                    CoveragePublicationScorer(aggregation_config.config, reference_database)
                )
            elif isinstance(aggregation_config.config, RecencyPublicationScorerConfig):
                self.scorers.append(
                    RecencyPublicationScorer(aggregation_config.config, reference_database)
                )
            else:
                raise ConfigurationError(f"Unsupported aggregation config: {aggregation_config}")

    def score(
        self, publication_concept_links: "Sequence[PublicationConceptLink]"
    ) -> dict["PublicationId", float]:
        aggregated_score_dict: dict[PublicationId, float] = {}
        for scorer, weight in zip(self.scorers, self.weights, strict=True):
            score_dict = scorer.score(publication_concept_links)
            for publication_id, score in score_dict.items():
                aggregated_score = aggregated_score_dict.get(publication_id, 0.0) + weight * score
                aggregated_score_dict[publication_id] = aggregated_score

        return aggregated_score_dict
