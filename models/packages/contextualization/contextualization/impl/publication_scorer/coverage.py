from collections import defaultdict
from typing import TYPE_CHECKING

from contextualization.core.contracts.publication_scorer import PublicationScorer
from contextualization.core.entities.config.publication_scorer import (
    CoveragePublicationScorerConfig,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from contextualization.core.entities.link import PublicationConceptLink
    from contextualization.core.entities.publication import PublicationId


class CoveragePublicationScorer(PublicationScorer[CoveragePublicationScorerConfig]):
    def _numbers_of_concepts_for_publications(
        self, publication_ids: "Sequence[PublicationId]"
    ) -> dict["PublicationId", float]:
        links_list = self.reference_database.get_publication_concept_links_from_publication_ids(
            publication_ids, flatten=False
        )
        return {
            publication_id: float(len({link.concept_id for link in links}))
            for publication_id, links in zip(publication_ids, links_list, strict=True)
        }

    def score(
        self, publication_concept_links: "Sequence[PublicationConceptLink]"
    ) -> dict["PublicationId", float]:
        numerator_dict: dict[PublicationId, float] = defaultdict(lambda: 0.0)
        for publication_concept_link in publication_concept_links:
            numerator_dict[publication_concept_link.publication_id] += 1

        publication_ids = list(numerator_dict.keys())

        denominator_dict = self._numbers_of_concepts_for_publications(publication_ids)

        score_dict: dict[PublicationId, float] = {}
        for publication_id in publication_ids:
            if denominator_dict[publication_id] > 0:
                score_dict[publication_id] = (
                    numerator_dict[publication_id] / denominator_dict[publication_id]
                )
            else:
                score_dict[publication_id] = 0

        return score_dict
