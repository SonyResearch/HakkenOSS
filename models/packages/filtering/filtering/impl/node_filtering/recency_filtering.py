import numpy as np

from filtering.core.contracts import KnowledgeGraph, NodeFiltering
from filtering.core.entities.candidate import InputNodeCandidate, OutputNodeCandidate
from filtering.core.entities.config.node_filtering import RecencyNodeFilteringConfig
from filtering.core.entities.kg import YearRange


class RecencyNodeFiltering(NodeFiltering[RecencyNodeFilteringConfig]):
    """Recency filtering model.
    This computes the filtering score of a node as follows.

    First, denoting the degree of the node across within the duration
    `year_range` by `D`.
    Then, split `year_range` into year windows of size `year_window_size`,
    obtaining `N = (year_range[1] - year_range[0]) // year_window_size` windows.
    Calculate the degree for each year window, and denote those by
    `delta[0], ..., delta[N-1]`.
    Also, for each year window, compute the weight linearly increased
    from `recency_min_weight` to `1`;
    i.e. `w[i] = i * (1 - recency_min_weight) / (N - 1) + recency_min_weight`.

    Then the score for the node is computed by
    `log(D + 1) * sum(w * delta)`, or equivalently `log(D * exp(sum(w * delta)))`,
    which exponentially gives advantage to recently made edges.
    """

    def __init__(self, config: RecencyNodeFilteringConfig, kg: KnowledgeGraph):
        super().__init__(config=config, kg=kg)

        if self.kg is None:
            raise ValueError("`kg` must be given when recency node filtering is used")
        self.kg: KnowledgeGraph

    def filter(
        self, candidates: list[InputNodeCandidate], max_output_candidates: int | None = None
    ) -> list[OutputNodeCandidate]:
        if not max_output_candidates:
            max_output_candidates = len(candidates)

        node_ids = [c.node_id for c in candidates]
        window_start_years = list(
            range(
                self.config.year_range.start,
                self.config.year_range.end - self.config.year_window_size,
                self.config.year_window_size,
            ),
        )
        window_degrees = [
            self.kg.get_degrees(
                node_ids=node_ids,
                direction=self.config.degree_direction,
                year_range=YearRange(
                    window_start_year, window_start_year + self.config.year_window_size
                ),
            )
            for window_start_year in window_start_years
        ]

        current_degree = self.kg.get_degrees(
            node_ids=node_ids,
            direction=self.config.degree_direction,
            year_range=self.config.year_range,
        )
        recency_weights = [
            i * (1 - self.config.recency_min_weight) / (len(window_degrees) - 1)
            + self.config.recency_min_weight
            for i in range(len(window_degrees))
        ]

        current_degree_arr = np.array(current_degree)
        window_degrees_arr = np.array(window_degrees)
        recency_weights_arr = np.array(recency_weights)

        scores_arr = np.log(current_degree_arr + 1) * np.sum(
            window_degrees_arr * recency_weights_arr[:, None]
        )
        scores = scores_arr.tolist()

        max_inds = np.argsort(scores)[::-1]

        outputs = []
        for i in max_inds[:max_output_candidates]:
            outputs.append(
                OutputNodeCandidate(node_id=candidates[i].node_id, filter_score=scores[i])
            )

        return outputs
