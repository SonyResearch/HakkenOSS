import numpy as np

from filtering.core.contracts import TripleFiltering
from filtering.core.entities.candidate import (
    InputTripleCandidate,
    OutputTripleCandidate,
)
from filtering.core.entities.config.triple_filtering import RandomTripleFilteringConfig


class RandomTripleFiltering(TripleFiltering[RandomTripleFilteringConfig]):
    def filter(
        self,
        candidates: list[InputTripleCandidate],
        max_output_candidates: int | None = None,
    ) -> list[OutputTripleCandidate]:
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
            output_cand = OutputTripleCandidate(
                symbol_mappings=cand.symbol_mappings, triple=cand.triple, filter_score=-1
            )
            filtered_candidates.append(output_cand)

        return filtered_candidates
