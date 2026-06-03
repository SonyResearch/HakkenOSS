import pandas as pd

from .base import ExplanationReranker


class ScoreReranker(ExplanationReranker):
    """Reranks explanations by score in descending order."""

    def rerank(self, explanations_df: pd.DataFrame) -> pd.DataFrame:
        return explanations_df.sort_values(by="score", ascending=False).reset_index()
