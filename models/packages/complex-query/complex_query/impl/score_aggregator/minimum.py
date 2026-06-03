from complex_query.core.contracts.score_aggregator import ScoreAggregator
from complex_query.core.entities.config.score_aggregator import MinimumScoreAggregatorConfig


class MinimumScoreAggregator(ScoreAggregator[MinimumScoreAggregatorConfig]):
    def binary_t_norm(self, a: float, b: float) -> float:
        return min(a, b)
