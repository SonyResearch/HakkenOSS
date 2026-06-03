import logging

import pandas as pd
from pydantic import BaseModel

from hypgen_pipeline.core.contracts.data_processor_base import DataProcessorBase
from hypgen_pipeline.core.values.defaults import (
    CONFIDENCE_SCORE_COLUMN_DEFAULT,
    CONFIDENCE_SCORES_COLUMN_DEFAULT,
    EXISTING_RELATIONS_COLUMN_DEFAULT,
    NODE_PAIR_COLUMN_DEFAULT,
    NODE_PAIR_OCIDS_COLUMN_DEFAULT,
    PREDICTED_RELATION_COLUMN_DEFAULT,
    PREDICTED_RELATIONS_COLUMN_DEFAULT,
)
from hypgen_pipeline.core.values.exceptions import InvalidColumnError

logger = logging.getLogger(__name__)


class DataProcessorPrepareConfig(BaseModel):
    node_pair_column: str = NODE_PAIR_COLUMN_DEFAULT
    node_pair_ocids_column: str = NODE_PAIR_OCIDS_COLUMN_DEFAULT
    predicted_relations_column: str = PREDICTED_RELATIONS_COLUMN_DEFAULT
    confidence_scores_column: str = CONFIDENCE_SCORES_COLUMN_DEFAULT
    existing_relations_column: str = EXISTING_RELATIONS_COLUMN_DEFAULT


class DataProcessorFinalizeConfig(BaseModel):
    sort_by_column: str = CONFIDENCE_SCORE_COLUMN_DEFAULT
    list_columns: list[str] | None = None
    format_token: str | None = None


class DataProcessor(DataProcessorBase[DataProcessorPrepareConfig, DataProcessorFinalizeConfig]):
    @staticmethod
    def prepare(df: pd.DataFrame, config: DataProcessorPrepareConfig) -> pd.DataFrame:
        """A method to prepare the raw output of the core model
        for the hypothesis generation filtering pipeline

        Args:
            df (pd.DataFrame): The dataframe with hypothesis
            config (DataProcessorPrepareConfig): a pydantic config that points to
                relevant parameters and columns of the dataframe for the processing

        Returns:
            pd.DataFrame: Processed and ready for subsequent steps
        """
        # collect relevant params
        node_pair_column = config.node_pair_column
        node_pair_ocids_column = config.node_pair_ocids_column
        predicted_relations_column = config.predicted_relations_column
        confidence_scores_column = config.confidence_scores_column
        existing_relations_column = config.existing_relations_column

        logger.info(f"Number of entities couples: {len(df)}")

        # Drop spurious index columns and all lines without any predictions
        logger.info("Droppig entities couples with no predicted hypothesis ...")
        df = df.loc[~df[predicted_relations_column].isna()].reset_index(drop=True)
        if "Unnamed: 0" in df.columns:
            df = df.drop("Unnamed: 0", axis=1)

        df[existing_relations_column] = df[existing_relations_column].fillna("")
        logger.info(f"Number of entities couples: {len(df)}")

        # Clean file
        df[node_pair_column] = df[node_pair_column].apply(
            lambda x: [elem.strip() for elem in x.split(" <=====> ")]
        )
        df[node_pair_ocids_column] = df[node_pair_ocids_column].apply(
            lambda x: [elem.strip() for elem in x.split(" <=====> ")]
        )
        df[predicted_relations_column] = df[predicted_relations_column].apply(
            lambda x: [elem.strip() for elem in x.split(", ")]
        )
        df[confidence_scores_column] = df[confidence_scores_column].apply(
            lambda x: [float(elem) for elem in x.split(", ")]
        )
        df[existing_relations_column] = df[existing_relations_column].apply(
            lambda x: [elem.strip() for elem in x.split(", ")]
        )

        # Explode dataframe to have one hypothesis per line
        df = df.explode([predicted_relations_column, confidence_scores_column]).reset_index(
            drop=True
        )
        logger.info(f"Number of hypothesis: {len(df)}")

        # Remove predictions that are already present in the graph
        logger.info("Removing predictions already present in the graph...")
        df = df.loc[
            ~df.apply(
                lambda x: x[predicted_relations_column] in x[existing_relations_column], axis=1
            )
        ].reset_index(drop=True)
        logger.info(f"Number of hypothesis: {len(df)}")

        selected_cols = [
            node_pair_column,
            node_pair_ocids_column,
            predicted_relations_column,
            confidence_scores_column,
        ]
        df = df[selected_cols]

        # Rename columns
        df = df.rename(
            {
                node_pair_column: NODE_PAIR_COLUMN_DEFAULT,
                node_pair_ocids_column: NODE_PAIR_OCIDS_COLUMN_DEFAULT,
                predicted_relations_column: PREDICTED_RELATION_COLUMN_DEFAULT,  # singular renaming
                confidence_scores_column: CONFIDENCE_SCORE_COLUMN_DEFAULT,  # singular renaming
            },
            axis=1,
        )

        # Add an idx
        return DataProcessor._generate_hypothesis_ids(df)

    @staticmethod
    def _generate_hypothesis_ids(df: pd.DataFrame) -> pd.DataFrame:
        columns = df.columns
        df["hypothesis_idx"] = [f"#{i}" for i in range(len(df))]
        ordered_columns = ["hypothesis_idx"]
        ordered_columns.extend(columns)
        logger.info("Generated a unique id for each hypothesis.")
        return df[ordered_columns]

    @staticmethod
    def finalize(df: pd.DataFrame, config: DataProcessorFinalizeConfig) -> pd.DataFrame:
        """A method to finalize the list of hypothesis in a specific format.
        Allows to decaouple this task from other filtering stages.

        Args:
            df (pd.DataFrame): The dataframe with hypothesis
            config (DataProcessorFinalizeConfig): a pydantic config that points to
                relevant parameters and columns of the dataframe for the finalization

        Returns:
            pd.DataFrame: A dataframe formatted, sorted, finalized for delivery.
        """
        sort_by_column = config.sort_by_column
        list_columns = config.list_columns
        format_token = config.format_token

        # Sort the hypothesis by the given criterium
        if sort_by_column not in df.columns:
            raise InvalidColumnError(
                message=f"Column '{sort_by_column}' does not exist in dataframe."
            )

        df = DataProcessor._sort_by(df, column=sort_by_column)

        # Format the list-like columns to be more readable
        return (
            DataProcessor._format_list_columns(df, columns=list_columns, format_token=format_token)
            if list_columns is not None
            else df
        )

    @staticmethod
    def _sort_by(df: pd.DataFrame, column: str):
        return df.sort_values(by=column, ascending=False)

    @staticmethod
    def _format_list_columns(df: pd.DataFrame, columns: list[str], format_token: str | None = None):
        for col in columns:
            if col not in df.columns:
                logger.info(f"Column '{col}' not present in dataframe, skipping formatting.")
            else:
                df[col] = df[col].apply(
                    lambda x: f"{x[0]} {format_token} {x[1]}"
                    if format_token is not None
                    else f"{x[0]} {x[1]}"
                )
        return df
