import logging
import sys
from pathlib import Path

import yaml
from hakken_ml_toolkit.ml_utils import DSVUtils, PickleUtils

from hypgen_pipeline.core.values.defaults import DELIMITER_DEFAULT
from hypgen_pipeline.impl.topk_filter import (
    CONFIDENCE_SCORE_COLUMN_DEFAULT,
    NODE_PAIR_COLUMN_DEFAULT,
    TOPK_COLUMN_DEFAULT,
    TopKFilter,
    TopKFilterConfig,
)
from hypgen_pipeline.utils.logging_utils import setup_logger

logger = logging.getLogger()


def main():
    setup_logger()

    if len(sys.argv) != 4:
        sys.stderr.write("Arguments error. Usage:\n\n")
        sys.stderr.write("filter-topk-entities input-file output-folder params-file\n")
        sys.exit(1)

    inputfile = Path(sys.argv[1])
    outputpath = Path(sys.argv[2])
    params_file = Path(sys.argv[3])

    # Collect params
    with open(params_file) as file:
        params = yaml.safe_load(file)
    params = params["filter_topk_entities"]

    node_pair_column = params.get("node_pair_column", NODE_PAIR_COLUMN_DEFAULT)
    confidence_score_column = params.get("confidence_score_column", CONFIDENCE_SCORE_COLUMN_DEFAULT)
    topk = params.get("topk", TOPK_COLUMN_DEFAULT)

    # Load
    df_hypothesis = PickleUtils.load(file_path=inputfile)
    logger.info("Loaded input file.")

    # Filter
    df_hypothesis = TopKFilter.filter(
        df=df_hypothesis,
        config=TopKFilterConfig(
            node_pair_column=node_pair_column,
            confidence_score_column=confidence_score_column,
            topk=topk,
        ),
    )
    logger.info("Topk filtering completed.")

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
