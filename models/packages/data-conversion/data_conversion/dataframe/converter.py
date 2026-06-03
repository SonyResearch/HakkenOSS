import json
import os
import warnings

import pandas as pd


class DataFrameConverter:
    @staticmethod
    def rows_to_json_records(df: pd.DataFrame, include_index: bool = False) -> list[str]:
        """Transforms a dataframe to a list of json records, where `list[i]` is
        row `i` represented in a json format."""
        json_records = []
        if include_index:
            if "index" in df.columns:
                warnings.warn(
                    "There is already a column named `index` in the dataframe, which can "
                    "cause confusion with the dataframe index.",
                    stacklevel=2,
                )
            df = df.reset_index()
        for _, row in df.iterrows():
            json_record = row.to_json()
            json_records.append(json_record)
        return json_records

    @staticmethod
    def save_rows_as_json_records(
        df: pd.DataFrame,
        output_dir: str,
        filename_template: str = "record_{index:d}.json",
        use_row_index_as_record_name: bool = False,
    ) -> None:
        """
        Saves each row of the dataframe as a separate .json file.
        The files are named incrementally from 0 to the number of rows, except if
         `use_row_index_as_record_name`, in which case the index of the row is the name
         of the json file.
        """
        # Create the folder if it does not exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Loop through the records and save each one with an incrementing name
        json_records = DataFrameConverter.rows_to_json_records(
            df, include_index=use_row_index_as_record_name
        )
        for i, json_record in enumerate(json_records):
            # Index record filename
            dict_record = json.loads(json_record)
            record_index = i if not use_row_index_as_record_name else dict_record["index"]
            filename = filename_template.format(index=record_index)
            # Save record to json
            filepath = os.path.join(output_dir, filename)
            with open(filepath, "w", encoding="utf-8") as file:
                if use_row_index_as_record_name:
                    del dict_record["index"]
                json.dump(dict_record, file)
