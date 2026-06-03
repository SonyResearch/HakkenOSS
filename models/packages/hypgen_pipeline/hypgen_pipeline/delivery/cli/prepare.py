import logging
import sys
from pathlib import Path

import yaml
from hakken_ml_toolkit.ml_utils import DSVUtils, PickleUtils

from hypgen_pipeline.core.values.defaults import (
    CONFIDENCE_SCORES_COLUMN_DEFAULT,
    DELIMITER_DEFAULT,
    EXISTING_RELATIONS_COLUMN_DEFAULT,
    NODE_PAIR_COLUMN_DEFAULT,
    NODE_PAIR_OCIDS_COLUMN_DEFAULT,
    PREDICTED_RELATIONS_COLUMN_DEFAULT,
)
from hypgen_pipeline.impl.data_processor import (
    DataProcessor,
    DataProcessorPrepareConfig,
)
from hypgen_pipeline.utils.logging_utils import setup_logger

logger = logging.getLogger()


def main():
    setup_logger()

    if len(sys.argv) != 4:
        sys.stderr.write("Arguments error. Usage:\n\n")
        sys.stderr.write("prepare input-file output-folder params-file\n")
        sys.exit(1)

    inputfile = Path(sys.argv[1])
    outputpath = Path(sys.argv[2])
    params_file = Path(sys.argv[3])

    # Collect params
    with open(params_file) as file:
        params = yaml.safe_load(file)
    params = params["prepare"]

    df_hypothesis = DSVUtils.read_dsv(file_path=inputfile, delimiter=DELIMITER_DEFAULT, header=0)
    logger.info("Loaded input file.")

    node_pair_column = params.get("node_pair_column", NODE_PAIR_COLUMN_DEFAULT)
    node_pair_ocids_column = params.get("node_pair_ocids_column", NODE_PAIR_OCIDS_COLUMN_DEFAULT)
    predicted_relations_column = params.get(
        "predicted_relations_column", PREDICTED_RELATIONS_COLUMN_DEFAULT
    )
    confidence_scores_column = params.get(
        "confidence_scores_column", CONFIDENCE_SCORES_COLUMN_DEFAULT
    )
    existing_relations_column = params.get(
        "existing_relations_column", EXISTING_RELATIONS_COLUMN_DEFAULT
    )

    # Load and Clean
    df_hypothesis = DataProcessor.prepare(
        df=df_hypothesis,
        config=DataProcessorPrepareConfig(
            node_pair_column=node_pair_column,
            node_pair_ocids_column=node_pair_ocids_column,
            predicted_relations_column=predicted_relations_column,
            confidence_scores_column=confidence_scores_column,
            existing_relations_column=existing_relations_column,
        ),
    )
    logger.info("Raw file preparation completed.")

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
