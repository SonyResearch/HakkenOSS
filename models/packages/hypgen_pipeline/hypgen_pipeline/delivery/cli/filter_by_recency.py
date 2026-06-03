import logging
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from hakken_ml_toolkit.ml_utils import DSVUtils, PickleUtils

from hypgen_pipeline.core.entities.kg import KgEdgesColumns
from hypgen_pipeline.core.values.defaults import DELIMITER_DEFAULT
from hypgen_pipeline.core.values.exceptions import MissingReferenceKgError
from hypgen_pipeline.impl.recency_filter import RecencyFilter, RecencyFilterConfig
from hypgen_pipeline.utils.logging_utils import setup_logger

load_dotenv()
logger = logging.getLogger()


def main():
    setup_logger()

    if len(sys.argv) != 4:
        sys.stderr.write("Arguments error. Usage:\n\n")
        sys.stderr.write("filter-by-recency input-file output-folder params-file\n")
        sys.exit(1)

    inputfile = Path(sys.argv[1])
    outputpath = Path(sys.argv[2])
    params_file = Path(sys.argv[3])

    # Collect params
    with open(params_file) as file:
        params = yaml.safe_load(file)
    params = params["filter_by_recency"]

    reference_kg_var = params.get("reference_kg_filepath_tsv", None)
    reference_kg_filepath = (
        os.environ.get(reference_kg_var) if reference_kg_var is not None else None
    )

    median_year = params.get("median_year", None)
    temporal_popularity_cache = params.get("temporal_popularity_cache", True)
    reference_kg_stats_var = params.get("reference_kg_stats_filepath_json", None)
    reference_kg_stats_filepath_json = os.environ.get(reference_kg_stats_var, "stats.json")
    reference_kg_stats_filepath_json = Path(reference_kg_stats_filepath_json)

    if reference_kg_filepath is None or not Path(reference_kg_filepath).is_file():
        MissingReferenceKgError(message="Provide a valid path for the reference kg!")

    # Load
    df_hypothesis = PickleUtils.load(file_path=inputfile)
    logger.info("Loaded input file.")
    df_kg = DSVUtils.read_dsv(
        file_path=reference_kg_filepath, delimiter=DELIMITER_DEFAULT, header=0
    )
    logger.info("Loaded reference kg.")

    # Compute temporal statistics
    logger.info("Computing temporal statistics from reference kg ...")
    entity_research_years, entities_papers_count = RecencyFilter.get_entity_temporal_popularity(
        graph_df=df_kg,
        kg_columns=KgEdgesColumns(),
        reference_kg_stats_filepath_json=reference_kg_stats_filepath_json,
        cache=temporal_popularity_cache,
    )
    logger.info("Computing temporal statistics from reference kg ... DONE")

    # Filter by recency
    config = RecencyFilterConfig(
        median_year=median_year,
        entities_research_year_statistics=entity_research_years,
        entities_papers_count=entities_papers_count,
    )
    df_hypothesis = RecencyFilter.filter(df=df_hypothesis, config=config)
    logger.info("Completed recency filtering.")

    logger.info(f"Number of remaining hypothesis: {len(df_hypothesis)}")

    # Save to pkl and csv
    logger.info(f"Saving files inside '{outputpath}'...")
    outfile = outputpath / "hypothesis.pkl"
    PickleUtils.save(data=df_hypothesis, file_path=outfile)

    outfile = outputpath / "hypothesis.tsv"
    DSVUtils.write_dsv(df=df_hypothesis, file_path=outfile, delimiter=DELIMITER_DEFAULT)

    logger.info("STEP completed.")


if __name__ == "__main__":
    main()
