import numpy as np

from filtering.core.contracts import KnowledgeGraph, NodeFiltering
from filtering.core.entities.candidate import InputNodeCandidate, OutputNodeCandidate
from filtering.core.entities.config.node_filtering import EntropyNodeFilteringConfig
from filtering.core.entities.kg import YearRange


class EntropyNodeFiltering(NodeFiltering[EntropyNodeFilteringConfig]):
    def __init__(self, config: EntropyNodeFilteringConfig, kg: KnowledgeGraph):
        super().__init__(config=config, kg=kg)

        if self.kg is None:
            raise ValueError("`kg` must be given when entropy node filtering is used")
        self.kg: KnowledgeGraph

    def filter(
        self, candidates: list[InputNodeCandidate], max_output_candidates: int | None = None
    ) -> list[OutputNodeCandidate]:
        if not max_output_candidates:
            max_output_candidates = len(candidates)

        node_ids = [c.node_id for c in candidates]
        degrees = np.array(
            self.kg.get_degrees(
                node_ids=node_ids,
                direction=self.config.degree_direction,
                year_range=self.config.year_range,
            )
        )
        degrees_start = np.array(
            self.kg.get_degrees(
                node_ids=node_ids,
                direction=self.config.degree_direction,
                year_range=YearRange(
                    self.config.year_range.start,
                    self.config.year_range.start + self.config.year_window_size,
                ),
            )
        )

        degrees_delta = degrees - degrees_start
        degrees_slope = degrees_delta / degrees
        degrees_slope = np.nan_to_num(degrees_slope)

        scores = (1 + degrees_slope) * np.log(degrees)
        max_inds = np.argsort(scores)[::-1]

        outputs = []
        for i in max_inds[:max_output_candidates]:
            outputs.append(
                OutputNodeCandidate(node_id=candidates[i].node_id, filter_score=scores[i])
            )

        return outputs
