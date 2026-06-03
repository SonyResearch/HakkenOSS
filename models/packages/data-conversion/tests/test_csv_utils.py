from __future__ import annotations

import csv
import os
import unittest

import pandas as pd

from data_conversion import CSVUtils


class CSVFactory:
    @staticmethod
    def create_sample_csv(filepath: str, rows: int) -> str:
        with open(filepath, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["col1", "col2", "col3"])
            for i in range(rows):
                writer.writerow([f"data{i}1", f"data{i}2", f"data{i}3"])
        return filepath


class TestCSVUtils(unittest.TestCase):
    def test_apply_function_to_csv_in_chunks(self):
        csv_file = CSVFactory.create_sample_csv("test_sample.csv", 20)
        chunk_size = 5

        # Example callable function that returns the sum of the lengths of the DataFrames
        # (number of rows processed)
        def example_callable(chunk: pd.DataFrame) -> int:
            return len(chunk)

        results = CSVUtils.apply_function_to_csv_in_chunks(csv_file, example_callable, chunk_size)

        self.assertEqual(
            len(results), 4
        )  # Since we have 20 rows and chunk size is 5, we should have 4 results
        self.assertTrue(
            all(res == chunk_size for res in results)
        )  # Each result should be equal to the chunk_size

        os.remove(csv_file)


if __name__ == "__main__":
    unittest.main()
