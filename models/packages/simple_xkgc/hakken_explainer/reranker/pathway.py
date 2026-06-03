import pandas as pd

from .base import ExplanationReranker


class PathwayReranker(ExplanationReranker):
    """Reranks explanations by prioritizing top-scoring explanation per pathway.

    Returns the highest-scoring explanation from each pathway first, followed by
    remaining explanations in descending score order.
    """

    def rerank(self, explanations_df: pd.DataFrame) -> pd.DataFrame:
        df = explanations_df.sort_values(by="score", ascending=False).reset_index()
        top_per_pathway = df.loc[df.groupby("pathway")["score"].idxmax()].sort_values(
            "score", ascending=False
        )

        remaining_rows = df[~df.index.isin(top_per_pathway.index)].sort_values(
            "score", ascending=False
        )

        return pd.concat([top_per_pathway, remaining_rows], ignore_index=True).reset_index()
