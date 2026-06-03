from abc import ABC, abstractmethod

import pandas as pd


class ExplanationReranker(ABC):
    """
    Base class for reranking explanations.
    """

    @abstractmethod
    def rerank(self, explanations_df: pd.DataFrame) -> pd.DataFrame:
        """
        Rerank the explanations.

        Args:
            explanations (pd.DataFrame): DataFrame containing explanations with
            columns 'pathway', 'explanation' and 'score'.

        Returns:
            pd.DataFrame: Reranked DataFrame.
        """
        pass
