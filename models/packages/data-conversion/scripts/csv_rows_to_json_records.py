from __future__ import annotations

import argparse
import json
import logging

from data_conversion import CSVUtils, DataFrameConverter

logging.getLogger().setLevel(logging.INFO)


class ParserArguments:
    csv_path: str
    output_dir: str
    dtype_json_path: str


def parse_args() -> ParserArguments:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--csv-path",
        type=str,
        required=True,
        help="Path to the `relations.csv` dataframe object from Digital Science.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to write the json records to.",
    )
    parser.add_argument(
        "--dtype-json-path",
        type=str,
        required=False,
        help="Path to a json file that indicates the csv type(s) for each of the csv columns."
        " Each key in the json is a column name and each value is the corresponding dtype."
        " This is given to the `dtype` kwarg of `pandas.read_csv`.",
    )
    args: ParserArguments = parser.parse_args()
    return args


def main(csv_path: str, output_dir: str, dtype_hints: dict[str, str] | None = None) -> None:
    CSVUtils.apply_function_to_csv_in_chunks(
        csv_path,
        DataFrameConverter.save_rows_as_json_records,
        chunk_size=10000,
        read_csv_kwargs={"sep": "\t", "dtype": dtype_hints},
        callable_kwargs={
            "output_dir": output_dir,
            "use_row_index_as_record_name": True,
        },
        show_progress_bar=True,
    )


if __name__ == "__main__":
    args = parse_args()
    with open(args.dtype_json_path, encoding="utf-8") as f:
        dtype_hints: dict[str, str] = json.load(f)
    main(args.csv_path, args.output_dir, dtype_hints)
