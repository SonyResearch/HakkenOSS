import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import median, mode

import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm

from hypgen_pipeline.core.contracts.entity_filter import FilterBase
from hypgen_pipeline.core.entities.kg import KgEdgesColumns
from hypgen_pipeline.core.values.defaults import (
    NODE_PAIR_OCIDS_COLUMN_DEFAULT,
)

logger = logging.getLogger(__name__)


class RecencyFilterConfig(BaseModel):
    median_year: int | None = None
    node_pair_ocids_column: str = NODE_PAIR_OCIDS_COLUMN_DEFAULT
    entities_research_year_statistics: dict[str, dict[str, int]] = {}
    entities_papers_count: dict[str, int] = {}


class RecencyFilter(FilterBase[RecencyFilterConfig]):
    """
    Filters hypotheses by selecting those where either one of the two entities
    has a median year (computed on the whole graph) that is above a certain threshold.
    """

    @staticmethod
    def filter(df: pd.DataFrame, config: RecencyFilterConfig) -> pd.DataFrame:
        nodes_pair_ocids_column = config.node_pair_ocids_column
        entity_research_year_stats = config.entities_research_year_statistics
        entities_papers_count = config.entities_papers_count

        # Add information with the recency mode and median to the dataframe
        df["recency_mode"] = df.apply(
            lambda x: [
                entity_research_year_stats[x[nodes_pair_ocids_column][0]]["mode"],
                entity_research_year_stats[x[nodes_pair_ocids_column][1]]["mode"],
            ],
            axis=1,
        )
        df["recency_median"] = df.apply(
            lambda x: [
                entity_research_year_stats[x[nodes_pair_ocids_column][0]]["median"],
                entity_research_year_stats[x[nodes_pair_ocids_column][1]]["median"],
            ],
            axis=1,
        )
        df["papers_count"] = df.apply(
            lambda x: [
                entities_papers_count[x[nodes_pair_ocids_column][0]],
                entities_papers_count[x[nodes_pair_ocids_column][1]],
            ],
            axis=1,
        )

        # If a median year is provided the filtering is performed
        if config.median_year is not None:

            def has_triplet_higher_median(x):
                median_subject = entity_research_year_stats[x[0]]["median"]
                median_object = entity_research_year_stats[x[1]]["median"]
                return median_subject >= config.median_year or median_object >= config.median_year

            df_filter = df.apply(
                lambda x: has_triplet_higher_median(x[nodes_pair_ocids_column]), axis=1
            )

            df = df[df_filter].reset_index(drop=True)

        return df

    @staticmethod
    def get_entity_temporal_popularity(
        graph_df: pd.DataFrame,
        kg_columns: KgEdgesColumns,
        reference_kg_stats_filepath_json: Path,
        cache: bool = True,
    ) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]]]:
        """Extracts and processes temporal information from an edges dataframe
        representing a knowledge graph.

        This function extracts temporal information for entities (objects and subjects),
        calculates their research years' statistics, and counts the number of papers
        associated with each entity.

        Args:
            graph_df : pd.DataFrame
                A pandas dataframe containing all the entities and edges in the graph
            kg_columns: KgEdgesColumns
                The columns found in the file containing the reference kg, including:
                    date_column : The column where the temporal information for an
                        edge is stored.
                    paper_link_column: The column where the link of the publication
                        related to that triple is stored
                    ocid_subject_column: The column with the id of the subject
                    ocid_object_column: The column with the id of the object
            reference_kg_stats_filepath_json: Path
                The path where the precomputed statistics can be found
            cache: bool
                If set to False, every statistic is recalculated, else is loaded from a file
                if it exists.
        Returns:
            tuple[dict[str, int], dict[str, int]]: two dictionaries as following
                1. entity_research_years : entities as keys and their research year statistics
                    (mode and median) as values.
                2. entity_papers : entities as keys and the count of associated papers as values.
        """
        output_json = reference_kg_stats_filepath_json

        if cache and output_json.exists():
            logger.info(f"Found existing file, loading '{output_json}'")
            with open(output_json) as f:
                entity_research_years, entity_papers = json.load(f)

        else:
            entity_research_years = defaultdict(list)
            entity_papers = defaultdict(set)
            invalid_time_rows = 0
            for _, item in tqdm(graph_df.iterrows()):
                date = item[kg_columns.date_column]
                ocid_subject = str(item[kg_columns.ocid_subject_column])
                ocid_object = str(item[kg_columns.ocid_object_column])
                paper_link = item[kg_columns.paper_link_column]

                try:
                    # Extract the year
                    year = datetime.fromtimestamp(date).year
                    entity_research_years[ocid_subject].append(year)
                    entity_research_years[ocid_object].append(year)

                    # Extract the publication
                    entity_papers[ocid_object].add(paper_link)
                    entity_papers[ocid_subject].add(paper_link)

                except ValueError:
                    invalid_time_rows += 1
                except TypeError:
                    invalid_time_rows += 1

            logger.info(f"Number of rows with invalid time information: {invalid_time_rows}")

            # Calculate time statistics
            for k, v in entity_research_years.items():
                entity_research_years[k] = {"mode": int(mode(v)), "median": int(median(v))}

            # Count number of publications
            for k, v in entity_papers.items():
                entity_papers[k] = len(v)

        if cache and not output_json.exists():
            logger.info(f"Saving dictionaries to '{output_json}'")
            with open(output_json, "w") as f:
                json.dump([entity_research_years, entity_papers], f, indent=4)

        return entity_research_years, entity_papers
