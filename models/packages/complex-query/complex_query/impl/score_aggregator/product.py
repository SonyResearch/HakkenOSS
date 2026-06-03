from complex_query.core.contracts.score_aggregator import ScoreAggregator
from complex_query.core.entities.config.score_aggregator import ProductScoreAggregatorConfig


class ProductScoreAggregator(ScoreAggregator[ProductScoreAggregatorConfig]):
    def binary_t_norm(self, a: float, b: float) -> float:
        return a * b
