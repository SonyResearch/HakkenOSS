import json
from datetime import datetime
from typing import cast

from loguru import logger

from data_processing.data_processor.processor_base import DataProcessor
from data_processing.utils.common import convert_list_to_string
from data_processing.utils.errors import ColumnContainsNullError
from data_processing.utils.hashing import hash_string
from data_processing.utils.pubtator import fetch_pubtator_data
from data_processing.values import (
    DOMAIN_PIPE_OBJECT_ID_COLUMN,
    DOMAIN_PIPE_SUBJECT_ID_COLUMN,
    NUMBER_OF_OCCURRENCES_COLUMN,
    OBJECT_DOMAIN_COLUMN,
    OBJECT_ID_COLUMN,
    OBJECT_ID_RAW_COLUMN,
    PMID_COLUMN,
    PMIDS_COLUMN,
    RELATION_ID_COLUMN,
    RELATION_TYPE_COLUMN,
    SUBJECT_DOMAIN_COLUMN,
    SUBJECT_ID_COLUMN,
    SUBJECT_ID_RAW_COLUMN,
    TIMESTAMP_COLUMN,
    YEAR_OCCURRENCES_COLUMN,
    DataFrameType,
)
from data_processing.values_pubtator import (
    PUBTATOR_PUBLICATION_METADATA_CACHE_FILE,
)


class PubtatorProcessor(DataProcessor[DataFrameType]):
    """Processor for Pubtator dataset with specific cleaning strategies"""

    def process(self) -> DataFrameType:
        """Pubtator-specific preparation strategies"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        # Load data into the dataframe self.df
        relations_cfg = self.config.data_files.relations[0]
        self.df = self.load_data(relations_cfg)
        count = self.adapter.count_rows(self.df)
        logger.info(f"Number of rows: {count}")

        # Step 1: Perform initial cleaning to simplify dataset
        self._clean()
        self.df = self.adapter.materialize_data(self.df, "clean", timestamp)
        self.adapter.show_head(self.df)
        count = self.adapter.count_rows(self.df)
        logger.info(f"Number of rows: {count}")

        # Step 2: Generate Entities Ids
        self._generate_entity_id()
        self.df = self.adapter.materialize_data(self.df, "entities_ids", timestamp)
        self.adapter.show_head(self.df)
        count = self.adapter.count_rows(self.df)
        logger.info(f"Number of rows: {count}")

        # Step 3: Format columns
        self._format_columns()
        self.df = self.adapter.materialize_data(self.df, "format_columns", timestamp)
        self.adapter.show_head(self.df)
        count = self.adapter.count_rows(self.df)
        logger.info(f"Number of rows: {count}")

        # Step 4: Create relation ids
        self._generate_relation_id()
        self.df = self.adapter.materialize_data(self.df, "relations_ids", timestamp)
        self.adapter.show_head(self.df)
        count = self.adapter.count_rows(self.df)
        logger.info(f"Number of rows: {count}")

        # Step 5: Fetch and Cache API info for each pmid
        unique_pmids = self._get_unique_pmids()
        logger.info(f"Sample of unique pmids:\n{unique_pmids[0:10]}")
        fetch_pubtator_data(unique_pmids)

        # Step 6: Add year and remove triples with missing time info
        self._get_publication_year()
        self.df = self.adapter.materialize_data(self.df, "publication_year", timestamp)
        count = self.adapter.count_rows(self.df)
        logger.info(f"Number of rows: {count}")

        # Step 7: Remove time duplicates and add occurrence info
        self._aggregate_triples_across_years()
        self.df = self.adapter.materialize_data(self.df, "triples_aggregated", timestamp)
        count = self.adapter.count_rows(self.df)
        logger.info(f"Number of rows after aggregation: {count}")

        # Step 8: Process sentences
        # Check that sentence_id and sentence do not repeat

        # Step 9: Process names
        # Check which name/mention is better AND the list of names for each id + domain with counts
        # in the sentence

        # Step 10: [optional] Process Licences fro ncbi API

        # Step 11: [optional] fetch authors and citation counts from ncbi API

        return self.df

    def _clean(self) -> None:
        logger.info("Dropping duplicates across all columns...")
        self.df = self.adapter.drop_duplicates(self.df)
        self.adapter.count_rows(self.df)

        logger.info("Cleaning natural text fields...")
        self.df = self.adapter.replace_pattern(
            self.df,
            columns=[
                RELATION_TYPE_COLUMN,
                DOMAIN_PIPE_SUBJECT_ID_COLUMN,
                DOMAIN_PIPE_OBJECT_ID_COLUMN,
            ],
            pattern=r'["\\]',
        )

        separator = r"\|"
        logger.info(f"Split columns with '{separator}' separator ...")
        self.df = self.adapter.split_column(
            self.df,
            column_name=DOMAIN_PIPE_SUBJECT_ID_COLUMN,
            column1_name=SUBJECT_DOMAIN_COLUMN,
            column2_name=SUBJECT_ID_RAW_COLUMN,
            separator=separator,
        )
        self.df = self.adapter.split_column(
            self.df,
            column_name=DOMAIN_PIPE_OBJECT_ID_COLUMN,
            column1_name=OBJECT_DOMAIN_COLUMN,
            column2_name=OBJECT_ID_RAW_COLUMN,
            separator=separator,
        )

        logger.info("Dropping rows with NaN ...")
        self.df = self.adapter.dropna(
            self.df,
            subset=[
                SUBJECT_DOMAIN_COLUMN,
                SUBJECT_ID_RAW_COLUMN,
                OBJECT_DOMAIN_COLUMN,
                OBJECT_ID_RAW_COLUMN,
                RELATION_TYPE_COLUMN,
                PMID_COLUMN,
            ],
        )
        self.adapter.count_rows(self.df)

        logger.info("Dropping entities, domains or relations with '-' as name ...")
        self.df = self.adapter.filter_by_value(
            self.df,
            columns=[
                SUBJECT_DOMAIN_COLUMN,
                SUBJECT_ID_RAW_COLUMN,
                OBJECT_DOMAIN_COLUMN,
                OBJECT_ID_RAW_COLUMN,
                RELATION_TYPE_COLUMN,
            ],
            invalid_value="-",
        )

    def _generate_entity_id(self) -> None:
        logger.info("Generating entieties ids through hashing...")
        self.df = self.adapter.apply_function_to_column(
            self.df, DOMAIN_PIPE_SUBJECT_ID_COLUMN, SUBJECT_ID_COLUMN, hash_string
        )
        self.df = self.adapter.apply_function_to_column(
            self.df, DOMAIN_PIPE_OBJECT_ID_COLUMN, OBJECT_ID_COLUMN, hash_string
        )
        self._check_for_collisions()

    def _check_for_collisions(self) -> None:
        logger.info("Checking for hash collisions across all IDs...")

        # Select relevant columns
        subjects_df = self.adapter.select_columns(
            self.df, [SUBJECT_ID_COLUMN, DOMAIN_PIPE_SUBJECT_ID_COLUMN]
        )
        objects_df = self.adapter.select_columns(
            self.df, [OBJECT_ID_COLUMN, DOMAIN_PIPE_OBJECT_ID_COLUMN]
        )

        # Rename columns
        subjects_df = self.adapter.rename_columns(
            subjects_df,
            {
                SUBJECT_ID_COLUMN: "hashed_id",
                DOMAIN_PIPE_SUBJECT_ID_COLUMN: "original_id",
            },
        )
        objects_df = self.adapter.rename_columns(
            objects_df,
            {
                OBJECT_ID_COLUMN: "hashed_id",
                DOMAIN_PIPE_OBJECT_ID_COLUMN: "original_id",
            },
        )

        all_ids_df = self.adapter.union(subjects_df, objects_df)

        # Group and aggregate
        collisions = self.adapter.groupby_aggregate(
            all_ids_df,
            groupby_cols=["hashed_id"],
            aggregations={"original_id": "count"},
        )

        collisions = self.adapter.filter_by_condition(collisions, condition="original_id_count > 1")

        collision_count = self.adapter.count_rows(collisions)

        if collision_count > 0:
            logger.error(f"Hash collisions detected! Found: {collision_count}")
            self.adapter.show_head(collisions)
            raise ValueError
        logger.info("✓ No hash collisions detected across all IDs!")

    def _format_columns(self) -> None:
        logger.info("Formatting columns ...")
        self.df = self.adapter.apply_function_to_column(
            self.df, OBJECT_DOMAIN_COLUMN, OBJECT_DOMAIN_COLUMN, str.upper
        )
        self.df = self.adapter.apply_function_to_column(
            self.df, SUBJECT_DOMAIN_COLUMN, SUBJECT_DOMAIN_COLUMN, str.upper
        )
        self.df = self.adapter.apply_function_to_column(
            self.df, RELATION_TYPE_COLUMN, RELATION_TYPE_COLUMN, str.upper
        )

    def _generate_relation_id(self) -> None:
        # TODO: check also here for collisions?
        logger.info("Generating relations ids through hashing...")

        # Concatenate relevant columns
        tmp_relation_id_column = f"{RELATION_ID_COLUMN}_tmp"
        self.df = self.adapter.concatenate_columns(
            self.df,
            [SUBJECT_DOMAIN_COLUMN, RELATION_TYPE_COLUMN, OBJECT_DOMAIN_COLUMN],
            tmp_relation_id_column,
            r"\|",
        )

        # Create hashed ids
        self.df = self.adapter.apply_function_to_column(
            self.df, tmp_relation_id_column, RELATION_ID_COLUMN, hash_string
        )

    def _get_unique_pmids(self) -> list[int]:
        return cast("list[int]", self.adapter.unique_values(self.df, PMID_COLUMN))

    def _get_publication_year(self) -> None:
        # Collect a map of pmid -> year
        with open(PUBTATOR_PUBLICATION_METADATA_CACHE_FILE) as f:
            year_map = {
                obj["pmid"]: obj["year"] for line in f if line.strip() for obj in [json.loads(line)]
            }

        # Map the pmids in the triples to the year
        def get_year(pmid: int) -> int:
            year = year_map.get(str(pmid))
            return int(year) if year else None

        self.df = self.adapter.apply_function_to_column(
            self.df, PMID_COLUMN, TIMESTAMP_COLUMN, get_year
        )

        # Remove triples without a year
        self.df = self.adapter.dropna(self.df, subset=[TIMESTAMP_COLUMN])

    def _aggregate_triples_across_years(self) -> None:
        """Aggregate multiple triples into unique edges with year and pmid metadata."""
        self.df = self.adapter.drop_duplicates(
            self.df,
            subset=[
                SUBJECT_ID_COLUMN,
                RELATION_TYPE_COLUMN,
                OBJECT_ID_COLUMN,
                TIMESTAMP_COLUMN,
                PMID_COLUMN,
            ],
        )
        groupby_cols = [SUBJECT_ID_COLUMN, RELATION_TYPE_COLUMN, OBJECT_ID_COLUMN]

        aggregations = {
            TIMESTAMP_COLUMN: ["min", "list"],
            PMID_COLUMN: "list",
            SUBJECT_DOMAIN_COLUMN: "first",
            OBJECT_DOMAIN_COLUMN: "first",
            SUBJECT_ID_RAW_COLUMN: "first",
            OBJECT_ID_RAW_COLUMN: "first",
            RELATION_ID_COLUMN: "first",
        }

        # Perform the aggregation
        self.df = self.adapter.groupby_aggregate(self.df, groupby_cols, aggregations)

        # Rename columns
        self.df = self.adapter.rename_columns(
            self.df,
            {
                f"{TIMESTAMP_COLUMN}_min": TIMESTAMP_COLUMN,
                f"{PMID_COLUMN}_list": PMIDS_COLUMN,
                f"{SUBJECT_DOMAIN_COLUMN}_first": SUBJECT_DOMAIN_COLUMN,
                f"{OBJECT_DOMAIN_COLUMN}_first": OBJECT_DOMAIN_COLUMN,
                f"{SUBJECT_ID_RAW_COLUMN}_first": SUBJECT_ID_RAW_COLUMN,
                f"{OBJECT_ID_RAW_COLUMN}_first": OBJECT_ID_RAW_COLUMN,
                f"{RELATION_ID_COLUMN}_first": RELATION_ID_COLUMN,
            },
        )

        # number_of_occurrences = size of year_occurrences.
        # Counts also same triple but different pmid in same year
        self.df = self.adapter.apply_function_to_column(
            self.df,
            f"{TIMESTAMP_COLUMN}_list",
            NUMBER_OF_OCCURRENCES_COLUMN,
            lambda years: len(years),
        )

        self.df = self.adapter.apply_function_to_column(
            self.df,
            f"{TIMESTAMP_COLUMN}_list",
            YEAR_OCCURRENCES_COLUMN,
            lambda years: sorted(set(years)),
        )

        # Drop the original collect_list column
        self.df = self.adapter.drop_columns(self.df, [f"{TIMESTAMP_COLUMN}_list"])

        # Convert year_occurrences and pmids into a string
        self.df = self.adapter.rename_columns(
            self.df,
            {
                YEAR_OCCURRENCES_COLUMN: f"{YEAR_OCCURRENCES_COLUMN}_tmp",
                PMIDS_COLUMN: f"{PMIDS_COLUMN}_tmp",
            },
        )  # needed for materialization in pyspark

        self.df = self.adapter.apply_function_to_column(
            self.df,
            f"{YEAR_OCCURRENCES_COLUMN}_tmp",
            YEAR_OCCURRENCES_COLUMN,
            lambda years: convert_list_to_string(years) if years else None,
        )

        self.df = self.adapter.apply_function_to_column(
            self.df,
            f"{PMIDS_COLUMN}_tmp",
            PMIDS_COLUMN,
            lambda pmids: convert_list_to_string(pmids) if pmids else None,
        )
        self.df = self.adapter.drop_columns(
            self.df, [f"{PMIDS_COLUMN}_tmp", f"{YEAR_OCCURRENCES_COLUMN}_tmp"]
        )

        # Note: if processed with pandas nulls will be removed. But at this point there should be
        # no nulls
        if self.adapter.column_has_nulls(self.df, YEAR_OCCURRENCES_COLUMN):
            raise ColumnContainsNullError(YEAR_OCCURRENCES_COLUMN)

        if self.adapter.column_has_nulls(self.df, PMIDS_COLUMN):
            raise ColumnContainsNullError(PMIDS_COLUMN)
