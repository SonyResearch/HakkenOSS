import numpy as np

from filtering.core.contracts import NodeFiltering
from filtering.core.entities.candidate import InputNodeCandidate, OutputNodeCandidate
from filtering.core.entities.config.node_filtering import RandomNodeFilteringConfig


class RandomNodeFiltering(NodeFiltering[RandomNodeFilteringConfig]):
    def filter(
        self, candidates: list[InputNodeCandidate], max_output_candidates: int | None = None
    ) -> list[OutputNodeCandidate]:
        if not max_output_candidates:
            max_output_candidates = len(candidates)

        random_number_generator = np.random.default_rng(seed=self.config.random_seed)
        random_indices = np.sort(
            random_number_generator.choice(
                len(candidates), size=max_output_candidates, replace=False
            )
        )

        filtered_candidates = []
        for i in random_indices:
            cand = candidates[i]
            output_cand = OutputNodeCandidate(node_id=cand.node_id, filter_score=-1)
            filtered_candidates.append(output_cand)

        return filtered_candidates
