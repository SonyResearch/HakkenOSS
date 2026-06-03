import logging
import sys
from pathlib import Path

import yaml
from hakken_ml_toolkit.ml_utils import DSVUtils, PickleUtils

from hypgen_pipeline.core.values.defaults import (
    CONFIDENCE_SCORE_COLUMN_DEFAULT,
    DELIMITER_DEFAULT,
    FORMAT_TOKEN_DEFAULT,
    NODE_PAIR_COLUMN_DEFAULT,
    NODE_PAIR_OCIDS_COLUMN_DEFAULT,
    PAPERS_COUNT_COLUMN_DEFAULT,
    RECENCY_MEDIAN_COLUMN_DEFAULT,
    RECENCY_MODE_COLUMN_DEFAULT,
)
from hypgen_pipeline.impl.data_processor import (
    DataProcessor,
    DataProcessorFinalizeConfig,
)
from hypgen_pipeline.utils.logging_utils import setup_logger

logger = logging.getLogger()


def main():
    setup_logger()

    if len(sys.argv) != 4:
        sys.stderr.write("Arguments error. Usage:\n\n")
        sys.stderr.write("finalize input-file output-folder params-file\n")
        sys.exit(1)

    inputfile = Path(sys.argv[1])
    outputpath = Path(sys.argv[2])
    params_file = Path(sys.argv[3])

    # Collect params
    with open(params_file) as file:
        params = yaml.safe_load(file)
    params = params["finalize"]

    df_hypothesis = PickleUtils.load(file_path=inputfile)
    logger.info("Loaded input file.")

    sort_by_column = params.get("sort_by_column", CONFIDENCE_SCORE_COLUMN_DEFAULT)
    format_token = params.get("format_token", FORMAT_TOKEN_DEFAULT)
    list_columns = params.get(
        "list_columns",
        [
            NODE_PAIR_COLUMN_DEFAULT,
            NODE_PAIR_OCIDS_COLUMN_DEFAULT,
            RECENCY_MEDIAN_COLUMN_DEFAULT,
            RECENCY_MODE_COLUMN_DEFAULT,
            PAPERS_COUNT_COLUMN_DEFAULT,
        ],
    )

    # Load and Clean
    config = DataProcessorFinalizeConfig(
        sort_by_column=sort_by_column, list_columns=list_columns, format_token=format_token
    )
    df_hypothesis = DataProcessor.finalize(df=df_hypothesis, config=config)
    logger.info("File finalization completed.")

    # Save to pkl and csv
    logger.info(f"Saving files inside '{outputpath}'...")
    outfile = outputpath / "hypothesis.pkl"
    PickleUtils.save(data=df_hypothesis, file_path=outfile)

    outfile = outputpath / "hypothesis.tsv"
    DSVUtils.write_dsv(df=df_hypothesis, file_path=outfile, delimiter=DELIMITER_DEFAULT)

    logger.info("STEP completed.")


if __name__ == "__main__":
    main()
