import pandas as pd
from pydantic import BaseModel

from hypgen_pipeline.core.contracts.entity_filter import FilterBase
from hypgen_pipeline.core.values.defaults import (
    CONFIDENCE_SCORE_COLUMN_DEFAULT,
    NODE_PAIR_COLUMN_DEFAULT,
    TOPK_COLUMN_DEFAULT,
)


class TopKFilterConfig(BaseModel):
    topk: int = TOPK_COLUMN_DEFAULT
    node_pair_column: str = NODE_PAIR_COLUMN_DEFAULT
    confidence_score_column: str = CONFIDENCE_SCORE_COLUMN_DEFAULT


class TopKFilter(FilterBase[TopKFilterConfig]):
    """
    Filters hypotheses by only selecting the hypotheses with top 3 confidence scores if the
    entities appear too many times in the batch.
    """

    @staticmethod
    def filter(df: pd.DataFrame, config: TopKFilterConfig) -> pd.DataFrame:
        """
        Complete data processing pipeline.

        Args:
            df (pd.DataFrame): The dataframe containing the hypothesis
            config (Dict[str, Any]): config dictionary containing paprameters for
                topk filtering.

        Returns:
            pd.DataFrame: Processed and filtered DataFrame
        """

        topk: int = config.topk
        node_pair_column: str = config.node_pair_column
        confidence_score_column: str = config.confidence_score_column

        df = TopKFilter._split_node_pairs(df, node_pair_column=node_pair_column)
        return TopKFilter._topk_entity_filtering(
            df=df, topk=topk, confidence_score_column=confidence_score_column
        )

    @staticmethod
    def _split_node_pairs(df: pd.DataFrame, node_pair_column: str = "node_pair") -> pd.DataFrame:
        """Split the node_pair column into separate ones.

        Args:
            df (pd.DataFrame): A dataframe with a column containing both node pairs
            node_pair_column (str, optional): The name of the column where the node pair
                is contained. Defaults to "node_pair".

        Returns:
            pd.DataFrame: The same dataframe where the column with the node pair
                is separated in two different columns.
        """
        df[["node1", "node2"]] = df[node_pair_column].to_list()
        return df

    @staticmethod
    def _topk_entity_filtering(
        df: pd.DataFrame,
        topk: int = 3,
        confidence_score_column: str = CONFIDENCE_SCORE_COLUMN_DEFAULT,
    ) -> pd.DataFrame:
        """
        Filter hypotheses based on entity frequency and confidence scores.

        For entities appearing more than frequency_threshold times:
        - Keep only the top_n hypotheses with highest confidence scores
        For other entities:
        - Keep all hypotheses

        Args:
            top_n (int, optional): Number of top confidence scores to keep
            confidence_score_column (str): Column where the confidnce score for sorting is stored

        Returns:
            pd.DataFrame: Filtered DataFrame
        """
        # Create a dictionary storing all the relations for each node and keep only the topk
        list_of_nodes = sorted(set(df["node1"].unique()) | set(df["node2"].unique()))

        # Select topk for each node irrespective of it being a subject or object
        topk_relations = []
        for node in list_of_nodes:
            relations = df.loc[(df["node1"] == node) | (df["node2"] == node)]
            relations = relations.sort_values(confidence_score_column, ascending=False).head(n=topk)
            topk_relations.append(relations)
        df = pd.concat(topk_relations)

        # Remove duplicates
        df = df.drop_duplicates(["node1", "node2", confidence_score_column]).reset_index(drop=True)

        return df.drop("node2", axis=1).drop("node1", axis=1)
