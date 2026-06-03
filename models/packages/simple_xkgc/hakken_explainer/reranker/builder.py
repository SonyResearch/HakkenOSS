from typing import Protocol

from hakken_explainer.constants import RerankStrategy
from hakken_explainer.reranker.pathway import PathwayReranker
from hakken_explainer.reranker.score import ScoreReranker

from .base import ExplanationReranker


class RerankerBuilder(Protocol):
    """
    Base class for reranking explanations.
    """

    @staticmethod
    def build(strategy: RerankStrategy) -> ExplanationReranker:
        if strategy == RerankStrategy.SCORES:
            return ScoreReranker()
        if strategy == RerankStrategy.UNIQUE_PATHWAYS:
            return PathwayReranker()
        msg = f"Unknown reranking strategy: {strategy}"
        raise ValueError(msg)
