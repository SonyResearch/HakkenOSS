import json
import os
import shutil
import unittest

import pandas as pd

from data_conversion import DataFrameConverter


class DataFrameFactory:
    @staticmethod
    def create_sample_dataframe(rows: int) -> pd.DataFrame:
        data = {
            "col1": [f"data{i}1" for i in range(rows)],
            "col2": [f"data{i}2" for i in range(rows)],
            "col3": [f"data{i}3" for i in range(rows)],
        }
        return pd.DataFrame(data)


class TestDataFrameConverter(unittest.TestCase):
    def test_rows_to_json_records(self):
        df = DataFrameFactory.create_sample_dataframe(5)
        json_records = DataFrameConverter.rows_to_json_records(df)
        self.assertEqual(len(json_records), 5)
        json_records = DataFrameConverter.rows_to_json_records(df, include_index=True)
        for json_record in json_records:
            dict_record = json.loads(json_record)
            self.assertTrue("index" in dict_record)

    def test_save_rows_as_json_records(self):
        df = DataFrameFactory.create_sample_dataframe(3)
        # Indexing using a counter
        DataFrameConverter.save_rows_as_json_records(
            df, "tmp", filename_template="record_{index:d}.json"
        )
        with open(os.path.join("tmp", "record_0.json")) as reader:
            dict_record_0 = json.load(reader)
            dict_df_0 = df.iloc[0].to_dict()
            self.assertTrue(dict_df_0, dict_record_0)
        shutil.rmtree("tmp")

        # Indexing using dataframe index
        DataFrameConverter.save_rows_as_json_records(
            df,
            "tmp",
            filename_template="record_{index:d}.json",
            use_row_index_as_record_name=True,
        )
        with open(os.path.join("tmp", "record_0.json")) as reader:
            dict_record_0 = json.load(reader)
            dict_df_0 = df.iloc[0].to_dict()
            self.assertTrue(dict_df_0, dict_record_0)
        shutil.rmtree("tmp")
