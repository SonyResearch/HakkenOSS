from collections.abc import Sequence
from typing import TYPE_CHECKING

from scipy.stats import rankdata

from contextualization.core.contracts.publication_scorer import PublicationScorer
from contextualization.core.entities.config.publication_scorer import (
    RecencyPublicationScorerConfig,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from contextualization.core.entities.link import PublicationConceptLink
    from contextualization.core.entities.publication import PublicationId


class RecencyPublicationScorer(PublicationScorer[RecencyPublicationScorerConfig]):
    def score(
        self, publication_concept_links: "Sequence[PublicationConceptLink]"
    ) -> dict["PublicationId", float]:
        publication_ids = list({link.publication_id for link in publication_concept_links})
        publication_years = [
            pub.year for pub in self.reference_database.get_publications(publication_ids)
        ]

        unnormalized_scores: Sequence[float]

        if self.config.use_rank:
            # Compute ranks of years and scale those to the range of 0 to 1
            # to lessen the impact of outliers.
            ranks = rankdata(publication_years, method="average").tolist()
            min_rank = min(ranks)
            max_rank = max(ranks)
            denominator = max_rank - min_rank
            subtractor = min_rank
            unnormalized_scores = ranks
        else:
            # Scale minimum and maximum year values to the range of 0 to 1.
            min_year = min(publication_years)
            max_year = max(publication_years)
            denominator = max_year - min_year
            subtractor = min_year
            unnormalized_scores = publication_years

        if denominator == 0:
            scores = [0.0 for _ in range(len(unnormalized_scores))]
        else:
            scores = [(score - subtractor) / denominator for score in unnormalized_scores]

        return dict(zip(publication_ids, scores, strict=True))
