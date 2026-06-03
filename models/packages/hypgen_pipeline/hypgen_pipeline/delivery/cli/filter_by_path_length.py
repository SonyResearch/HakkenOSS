import logging
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from hakken_ml_toolkit.ml_utils import DSVUtils, PickleUtils
from hakken_ml_toolkit.ml_utils.networkx import NetworkXUtils, NetworkXUtilsConfig

from hypgen_pipeline.core.values.defaults import (
    DELIMITER_DEFAULT,
    NODE_PAIR_OCIDS_COLUMN_DEFAULT,
    OCID_OBJECT_COLUMN_DEFAULT,
    OCID_SUBJECT_COLUMN_DEFAULT,
    RELATION_TYPE_COLUMN_DEFAULT,
)
from hypgen_pipeline.core.values.exceptions import MissingReferenceKgError
from hypgen_pipeline.impl.path_length_filter import PathLengthFilter, PathLengthFilterConfig
from hypgen_pipeline.utils.logging_utils import setup_logger

logger = logging.getLogger()
load_dotenv()


def main():
    setup_logger()

    if len(sys.argv) != 4:
        sys.stderr.write("Arguments error. Usage:\n\n")
        sys.stderr.write("filter-by-path-length input-file output-folder params-file\n")
        sys.exit(1)

    inputfile = Path(sys.argv[1])
    outputpath = Path(sys.argv[2])
    params_file = Path(sys.argv[3])

    # Collect params
    with open(params_file) as file:
        params = yaml.safe_load(file)
    params = params["filter_by_path_length"]

    reference_kg_var = params.get("reference_kg_filepath_tsv", None)
    reference_kg_filepath = (
        os.environ.get(reference_kg_var) if reference_kg_var is not None else None
    )
    node_pair_ocids_column = params.get("node_pair_ocids_column", NODE_PAIR_OCIDS_COLUMN_DEFAULT)
    include_extrema = params.get("include_extrema", False)
    max_path_length = params.get("max_path_length", None)
    min_path_length = params.get("min_path_length", None)

    if reference_kg_filepath is None or not Path(reference_kg_filepath).is_file():
        MissingReferenceKgError(message="Provide a valid path for the reference kg!")

    # Load
    df_hypothesis = PickleUtils.load(file_path=inputfile)
    logger.info("Loaded input file.")
    df_kg = DSVUtils.read_dsv(
        file_path=reference_kg_filepath, delimiter=DELIMITER_DEFAULT, header=0
    )
    logger.info("Loaded reference kg.")
    nxutils_config = NetworkXUtilsConfig(
        source_column=OCID_SUBJECT_COLUMN_DEFAULT,
        target_column=OCID_OBJECT_COLUMN_DEFAULT,
        relation_column=RELATION_TYPE_COLUMN_DEFAULT,
        multiple_edges=True,
        directed=False,
    )
    kg_graph = NetworkXUtils.load_graph_from_pandas(df_kg, config=nxutils_config)
    logger.info("Created nx graph.")

    # Filter by shortest path
    config = PathLengthFilterConfig(
        node_pair_ocids_column=node_pair_ocids_column,
        reference_kg=kg_graph,
        min_path_length=min_path_length,
        max_path_length=max_path_length,
        include_extrema=include_extrema,
    )

    df_hypothesis = PathLengthFilter.filter(df=df_hypothesis, config=config)
    logger.info("Computed shortest path length.")

    # Show statistics
    value_counts_df = df_hypothesis["shortest_path_length"].value_counts().reset_index()
    value_counts_df.columns = ["path_length", "counts"]
    logger.info(f"Shortest path length occurrences:\n {value_counts_df}")

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
