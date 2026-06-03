import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# Assuming these imports from your project
from data_processing.adapters.impl.pandas_adapter import PandasAdapter
from data_processing.utils.errors import (
    ColumnCountMismatchError,
    EmptyListError,
    MissingColumnError,
)
from data_processing.values import DEFAULT_SEPARATOR, PROCESSOR_CACHE_PATH


class TestPandasAdapterReadCSV(unittest.TestCase):
    """Test PandasAdapter read_csv method"""

    @patch("pandas.read_csv")
    def test_read_csv_defaults(self, mock_read_csv):
        """Test reading CSV with default parameters"""
        mock_df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        mock_read_csv.return_value = mock_df

        result = PandasAdapter.read_csv(path="/local/data.csv")

        mock_read_csv.assert_called_once_with("/local/data.csv", sep=DEFAULT_SEPARATOR, header=0)
        pd.testing.assert_frame_equal(result, mock_df)

    @patch("pandas.read_csv")
    def test_read_csv_custom_separator(self, mock_read_csv):
        """Test reading CSV with custom separator"""
        mock_df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        mock_read_csv.return_value = mock_df

        result = PandasAdapter.read_csv(path="/data.csv", sep=",")

        mock_read_csv.assert_called_once_with("/data.csv", sep=",", header=0)
        pd.testing.assert_frame_equal(result, mock_df)

    @patch("pandas.read_csv")
    def test_read_csv_no_header(self, mock_read_csv):
        """Test reading CSV without header"""
        mock_df = pd.DataFrame({0: [1, 2], 1: [3, 4]})
        mock_read_csv.return_value = mock_df

        result = PandasAdapter.read_csv(path="/data.csv", header=None)

        mock_read_csv.assert_called_once_with("/data.csv", sep=DEFAULT_SEPARATOR, header=None)
        pd.testing.assert_frame_equal(result, mock_df)

    @patch("pandas.read_csv")
    def test_read_csv_no_header_with_provided_names(self, mock_read_csv):
        """Test reading CSV with given column names"""
        mock_df = pd.DataFrame({0: [1, 2], 1: [3, 4]})
        mock_read_csv.return_value = mock_df

        result = PandasAdapter.read_csv(
            path="/data.csv", header=None, column_names=["col1", "col2"]
        )

        mock_read_csv.assert_called_once_with("/data.csv", sep=DEFAULT_SEPARATOR, header=None)
        pd.testing.assert_frame_equal(result, mock_df)

    @patch("pandas.read_csv")
    def test_column_names_override_header(self, mock_read_csv):
        """Behavior: if names provided, they win over header=0"""
        mock_df = pd.DataFrame({"a": [1], "b": [2]})
        mock_read_csv.return_value = mock_df

        result = PandasAdapter.read_csv(
            path="/data.csv",
            header=0,
            column_names=["x", "y"],
        )

        assert list(result.columns) == ["x", "y"]

    @patch("pandas.read_csv")
    def test_no_validation_when_column_names_none(self, mock_read_csv):
        mock_df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        mock_read_csv.return_value = mock_df

        result = PandasAdapter.read_csv(path="/data.csv")

        mock_read_csv.assert_called_once()
        pd.testing.assert_frame_equal(result, mock_df)

    @patch("pandas.read_csv")
    def test_read_csv_no_header_with_provided_names_wrong_count(self, mock_read_csv):
        """Test reading CSV with given column names in incorrect amount"""
        mock_df = pd.DataFrame({0: [1, 2], 1: [3, 4]})
        mock_read_csv.return_value = mock_df

        with pytest.raises(ColumnCountMismatchError):
            PandasAdapter.read_csv(
                path="/data.csv", header=None, column_names=["col1", "col2", "col3"]
            )

        mock_df = pd.DataFrame({"hello": [1, 2], "morgen": [3, 4]})
        mock_read_csv.return_value = mock_df

        with pytest.raises(ColumnCountMismatchError):
            PandasAdapter.read_csv(
                path="/data.csv", header=True, column_names=["col1", "col2", "col3"]
            )


class TestPandasAdapterToCSV(unittest.TestCase):
    """Test PandasAdapter to_csv method"""

    def test_to_csv_defaults(self):
        """Test writing CSV with default parameters"""
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})

        tmpdir = Path(tempfile.mkdtemp())
        output_path = tmpdir / "data"
        with patch.object(df, "to_csv") as mock_to_csv:
            PandasAdapter.to_csv(df, str(output_path))

            mock_to_csv.assert_called_once_with(
                tmpdir / "data.tsv", sep=DEFAULT_SEPARATOR, index=False
            )

    def test_to_csv_creates_directory(self):
        """Test creating a nested dir that does not exist"""
        df = pd.DataFrame({"col1": [1, 2]})
        tmpdir = Path(tempfile.mkdtemp())
        output_path = tmpdir / "data" / "nested"

        with patch("pathlib.Path.mkdir") as mock_mkdir, patch.object(df, "to_csv") as mock_to_csv:
            PandasAdapter.to_csv(df, str(output_path))

            # Directory creation called
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

            # CSV write still called correctly
            mock_to_csv.assert_called_once()

    def test_to_csv_custom_separator(
        self,
    ):
        """Test writing CSV with custom separator"""
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})

        tmpdir = Path(tempfile.mkdtemp())
        output_path = tmpdir / "data"

        with patch.object(df, "to_csv") as mock_to_csv:
            PandasAdapter.to_csv(df, output_path, output_sep=",")

            mock_to_csv.assert_called_once_with(tmpdir / "data.csv", sep=",", index=False)

    def test_to_csv_with_index(self):
        """Test writing CSV with index"""
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})

        tmpdir = Path(tempfile.mkdtemp())
        output_path = tmpdir / "data"

        with patch.object(df, "to_csv") as mock_to_csv:
            PandasAdapter.to_csv(df, output_path, index=True)

            mock_to_csv.assert_called_once_with(
                tmpdir / "data.tsv", sep=DEFAULT_SEPARATOR, index=True
            )

    def test_to_csv_all_custom_parameters(self):
        """Test writing CSV with all custom parameters"""
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})

        tmpdir = Path(tempfile.mkdtemp())
        output_path = tmpdir / "data"

        with patch.object(df, "to_csv") as mock_to_csv:
            PandasAdapter.to_csv(df, output_path, output_sep="|", index=True)

            mock_to_csv.assert_called_once_with(tmpdir / "data.unknown", sep="|", index=True)


class TestPandasAdapterConcat(unittest.TestCase):
    """Test PandasAdapter concat method"""

    def test_concat(self):
        """Test concat"""
        df = pd.DataFrame({"col1": [1, 2, 4], "col2": [5, 7, 8]})

        result = PandasAdapter.concat([df, df])

        expected = pd.DataFrame(
            {"col1": [1, 2, 4, 1, 2, 4], "col2": [5, 7, 8, 5, 7, 8]}, index=[0, 1, 2, 0, 1, 2]
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_concat_empty_and_single_list(self):
        """Test concat empty and single element list"""
        # Single element list
        df = pd.DataFrame({"col1": [1, 2, 4], "col2": [5, 7, 8]})

        result = PandasAdapter.concat([df])

        expected = pd.DataFrame({"col1": [1, 2, 4], "col2": [5, 7, 8]}, index=[0, 1, 2])
        pd.testing.assert_frame_equal(result, expected)

        # Empty list
        with pytest.raises(EmptyListError):
            PandasAdapter.concat([])


class TestPandasAdapterDropNA(unittest.TestCase):
    """Test PandasAdapter dropna method"""

    def test_dropna_no_params(self):
        """Test dropna without parameters"""
        df = pd.DataFrame({"col1": [1, 2, np.nan, 4], "col2": [5, np.nan, 7, 8]})

        result = PandasAdapter.dropna(df)

        expected = pd.DataFrame({"col1": [1.0, 4.0], "col2": [5.0, 8.0]}, index=[0, 3])
        pd.testing.assert_frame_equal(result, expected)

    def test_dropna_with_subset(self):
        """Test dropna with subset parameter"""
        df = pd.DataFrame(
            {"col1": [1, np.nan, 3, 4], "col2": [5, 6, 7, 8], "col3": [9, 10, np.nan, 12]}
        )

        result = PandasAdapter.dropna(df, subset=["col1"])

        expected = pd.DataFrame(
            {"col1": [1.0, 3.0, 4.0], "col2": [5, 7, 8], "col3": [9.0, np.nan, 12.0]},
            index=[0, 2, 3],
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_dropna_with_axis(self):
        """Test dropna with axis parameter"""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [np.nan, np.nan, np.nan], "col3": [4, 5, 6]})

        result = PandasAdapter.dropna(df, axis=1)

        expected = pd.DataFrame({"col1": [1, 2, 3], "col3": [4, 5, 6]})
        pd.testing.assert_frame_equal(result, expected)

    def test_dropna_with_how(self):
        """Test dropna with how parameter"""
        df = pd.DataFrame({"col1": [np.nan, np.nan, 3], "col2": [np.nan, np.nan, 6]})

        result = PandasAdapter.dropna(df, how="all")

        expected = pd.DataFrame({"col1": [3.0], "col2": [6.0]}, index=[2])
        pd.testing.assert_frame_equal(result, expected)

    def test_dropna_with_thresh(self):
        """Test dropna with thresh parameter"""
        df = pd.DataFrame({"col1": [1, np.nan, 3], "col2": [np.nan, np.nan, 6], "col3": [7, 8, 9]})

        result = PandasAdapter.dropna(df, thresh=2)

        expected = pd.DataFrame(
            {"col1": [1.0, 3.0], "col2": [np.nan, 6.0], "col3": [7, 9]}, index=[0, 2]
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_dropna_multiple_params(self):
        """Test dropna with multiple parameters"""
        df = pd.DataFrame(
            {"id": [1, 2, np.nan, 4], "name": ["A", np.nan, "C", "D"], "value": [10, 20, 30, 40]}
        )

        result = PandasAdapter.dropna(df, subset=["id", "name"], how="any")

        expected = pd.DataFrame(
            {"id": [1.0, 4.0], "name": ["A", "D"], "value": [10, 40]}, index=[0, 3]
        )
        pd.testing.assert_frame_equal(result, expected)


class TestPandasAdapterDropDuplicates(unittest.TestCase):
    """Test PandasAdapter drop_duplicates method"""

    def test_drop_duplicates_no_params(self):
        """Test drop_duplicates without parameters"""
        df = pd.DataFrame({"col1": [1, 2, 1, 3], "col2": [4, 5, 4, 6]})

        result = PandasAdapter.drop_duplicates(df)

        expected = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]}, index=[0, 1, 3])
        pd.testing.assert_frame_equal(result, expected)

    def test_drop_duplicates_with_subset(self):
        """Test drop_duplicates with subset parameter"""
        df = pd.DataFrame({"id": [1, 2, 1, 3], "value": [10, 20, 30, 40]})

        result = PandasAdapter.drop_duplicates(df, subset=["id"])

        expected = pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 40]}, index=[0, 1, 3])
        pd.testing.assert_frame_equal(result, expected)

    def test_drop_duplicates_multiple_columns(self):
        """Test drop_duplicates with multiple columns"""
        df = pd.DataFrame(
            {
                "id": [1, 2, 1, 1],
                "email": ["a@b.com", "c@d.com", "a@b.com", "e@f.com"],
                "timestamp": ["2023-01-01", "2023-01-02", "2023-01-01", "2023-01-03"],
            }
        )

        result = PandasAdapter.drop_duplicates(df, subset=["id", "email", "timestamp"])

        expected = pd.DataFrame(
            {
                "id": [1, 2, 1],
                "email": ["a@b.com", "c@d.com", "e@f.com"],
                "timestamp": ["2023-01-01", "2023-01-02", "2023-01-03"],
            },
            index=[0, 1, 3],
        )
        pd.testing.assert_frame_equal(result, expected)


class TestPandasAdapterSplitColumn(unittest.TestCase):
    """Test PandasAdapter split_column method"""

    def test_split_column(self):
        """Test split_column without parameters"""

        df = pd.DataFrame({"full_name": ["Alice|Smith", "Bob|Johnson", "Charlie|Brown", "Dana"]})

        result_df = PandasAdapter.split_column(
            df,
            column_name="full_name",
            column1_name="first_name",
            column2_name="last_name",
            separator="\\|",
        )

        # Assert: check column creation
        self.assertIn("first_name", result_df.columns)
        self.assertIn("last_name", result_df.columns)

        # Assert: check correct splitting
        expected_first_names = ["Alice", "Bob", "Charlie", "Dana"]
        expected_last_names = ["Smith", "Johnson", "Brown", None]

        self.assertListEqual(result_df["first_name"].tolist(), expected_first_names)
        self.assertListEqual(result_df["last_name"].tolist(), expected_last_names)

        # Assert: DataFrame object returned is the same (modified in place)
        self.assertIs(result_df, df)


class TestPandasAdapterConcatenateColumns:
    def test_concatenate_columns_basic(self):
        df = pd.DataFrame({"col1": ["A", "B", "C"], "col2": ["X", "Y", "Z"]})
        result_df = PandasAdapter.concatenate_columns(df, ["col1", "col2"], "combined")
        expected = pd.DataFrame(
            {
                "col1": ["A", "B", "C"],
                "col2": ["X", "Y", "Z"],
                "combined": ["A|X", "B|Y", "C|Z"],
            }
        )
        pd.testing.assert_frame_equal(result_df, expected)

    def test_concatenate_columns_custom_delimiter(self):
        df = pd.DataFrame({"first": ["A", "B"], "second": ["1", "2"]})
        result_df = PandasAdapter.concatenate_columns(
            df, ["first", "second"], "joined", delimiter=","
        )
        expected = pd.DataFrame(
            {
                "first": ["A", "B"],
                "second": ["1", "2"],
                "joined": ["A,1", "B,2"],
            }
        )
        pd.testing.assert_frame_equal(result_df, expected)

    def test_concatenate_columns_missing_column_raises(self):
        df = pd.DataFrame({"col1": ["A"], "col2": ["X"]})
        with pytest.raises(MissingColumnError):
            PandasAdapter.concatenate_columns(df, ["col1", "col3"], "joined")


class TestPandasAdapterFilterByValue(unittest.TestCase):
    """Test PandasAdapter.filter_by_value method"""

    def test_filter_by_value(self):
        """Test filter_by_value removes rows with invalid values"""

        df = pd.DataFrame({"col1": ["A", "-", "C", "D"], "col2": ["X", "Y", "-", "Z"]})

        # Filter
        result_df = PandasAdapter.filter_by_value(df, columns=["col1", "col2"], invalid_value="-")

        # Assert: filtered rows should remove rows with '-' in col1 or col2
        expected_df = pd.DataFrame({"col1": ["A", "D"], "col2": ["X", "Z"]}).reset_index(drop=True)

        pd.testing.assert_frame_equal(result_df.reset_index(drop=True), expected_df)

        # Assert: DataFrame returned is a new object (not modified in place)
        self.assertIsNot(result_df, df)

    def test_filter_with_custom_invalid_value(self):
        """Test filter_by_value works with a custom invalid value"""
        df = pd.DataFrame({"col1": ["OK", "INVALID", "OK"], "col2": ["OK", "OK", "INVALID"]})

        result_df = PandasAdapter.filter_by_value(
            df, columns=["col1", "col2"], invalid_value="INVALID"
        )

        expected_df = pd.DataFrame({"col1": ["OK"], "col2": ["OK"]}).reset_index(drop=True)

        pd.testing.assert_frame_equal(result_df.reset_index(drop=True), expected_df)


class TestPandasAdapterFilterByCondition(unittest.TestCase):
    """Test PandasAdapter.filter_by_condition method"""

    def test_filter_by_simple_condition(self):
        """Test filtering rows with a simple numeric condition"""

        df = pd.DataFrame({"age": [25, 40, 18, 30], "name": ["Alice", "Bob", "Charlie", "Dana"]})

        # Filter
        result_df = PandasAdapter.filter_by_condition(df, "age >= 30")

        # Assert
        expected_df = pd.DataFrame({"age": [40, 30], "name": ["Bob", "Dana"]}).reset_index(
            drop=True
        )

        pd.testing.assert_frame_equal(result_df.reset_index(drop=True), expected_df)

    def test_filter_by_string_condition(self):
        """Test filtering rows with a string condition"""
        df = pd.DataFrame({"category": ["A", "B", "A", "C"], "value": [10, 20, 30, 40]})

        result_df = PandasAdapter.filter_by_condition(df, "category == 'A'")

        expected_df = pd.DataFrame({"category": ["A", "A"], "value": [10, 30]}).reset_index(
            drop=True
        )

        pd.testing.assert_frame_equal(result_df.reset_index(drop=True), expected_df)

    def test_returns_copy_not_view(self):
        """Test that the returned DataFrame is a copy, not a view"""
        df = pd.DataFrame({"x": [1, 2, 3, 4]})

        result_df = PandasAdapter.filter_by_condition(df, "x > 2")

        # Should not be the same object
        self.assertIsNot(result_df, df)

        # Modify result_df — original df should remain unchanged
        result_df.loc[:, "x"] = result_df["x"] * 10
        self.assertListEqual(df["x"].tolist(), [1, 2, 3, 4])


class TestPandasAdapterApplyFunctionToColumn:
    def test_apply_function_str_return(self):
        df = pd.DataFrame({"name": ["alice", "bob", "charlie"]})

        def to_upper(x: str) -> str:
            return x.upper()

        result_df = PandasAdapter.apply_function_to_column(df, "name", "upper_name", to_upper)
        expected = pd.DataFrame(
            {"name": ["alice", "bob", "charlie"], "upper_name": ["ALICE", "BOB", "CHARLIE"]}
        )
        pd.testing.assert_frame_equal(result_df, expected)

    def test_apply_function_int_return(self):
        df = pd.DataFrame({"value": [1, 2, 3]})

        def double(x: int) -> int:
            return x * 2

        result_df = PandasAdapter.apply_function_to_column(df, "value", "double_value", double)
        expected = pd.DataFrame({"value": [1, 2, 3], "double_value": [2, 4, 6]})
        pd.testing.assert_frame_equal(result_df, expected)

    def test_apply_function_float_return(self):
        df = pd.DataFrame({"value": [1, 2, 3]})

        def half(x: int) -> float:
            return x / 2.0

        result_df = PandasAdapter.apply_function_to_column(df, "value", "half_value", half)
        expected = pd.DataFrame({"value": [1, 2, 3], "half_value": [0.5, 1.0, 1.5]})
        pd.testing.assert_frame_equal(result_df, expected)

    def test_apply_function_bool_return(self):
        df = pd.DataFrame({"value": [1, 2, 3]})

        def is_even(x: int) -> bool:
            return x % 2 == 0

        result_df = PandasAdapter.apply_function_to_column(df, "value", "is_even", is_even)
        expected = pd.DataFrame({"value": [1, 2, 3], "is_even": [False, True, False]})
        pd.testing.assert_frame_equal(result_df, expected)

    def test_apply_function_no_type_hint(self):
        df = pd.DataFrame({"text": ["a", "bb", "ccc"]})

        def text_length(x):
            return len(x)

        result_df = PandasAdapter.apply_function_to_column(df, "text", "length", text_length)
        expected = pd.DataFrame({"text": ["a", "bb", "ccc"], "length": [1, 2, 3]})
        pd.testing.assert_frame_equal(result_df, expected)

    def test_apply_function_handles_empty_dataframe(self):
        """Test behavior when DataFrame is empty"""
        df = pd.DataFrame({"text": []})

        result_df = PandasAdapter.apply_function_to_column(
            df, column="text", new_column="processed", func=lambda x: x + "!"
        )

        expected_df = pd.DataFrame({"text": [], "processed": []})

        pd.testing.assert_frame_equal(result_df, expected_df)


class TestPandasAdapterSelectColumns(unittest.TestCase):
    """Test PandasAdapter.select_columns method"""

    def test_select_columns_basic(self):
        """Test selecting a subset of columns"""
        df = pd.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [4, 5, 6],
                "c": [7, 8, 9],
            }
        )

        result_df = PandasAdapter.select_columns(df, ["a", "c"])

        expected_df = pd.DataFrame(
            {
                "a": [1, 2, 3],
                "c": [7, 8, 9],
            }
        ).reset_index(drop=True)

        pd.testing.assert_frame_equal(result_df.reset_index(drop=True), expected_df)

    def test_select_columns_order_preserved(self):
        """Test that column order is preserved in the result"""
        df = pd.DataFrame(
            {
                "x": [1, 2],
                "y": [3, 4],
                "z": [5, 6],
            }
        )

        result_df = PandasAdapter.select_columns(df, ["z", "x"])
        self.assertListEqual(list(result_df.columns), ["z", "x"])

    def test_returns_copy_not_view(self):
        """Test that the returned DataFrame is a copy, not a view"""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
        df_original = df.copy()

        result_df = PandasAdapter.select_columns(df, ["a"])

        # Should not be the same object
        self.assertIsNot(result_df, df)

        # Modify result_df — should not affect df
        result_df.loc[:, "a"] = [100, 200, 300]
        pd.testing.assert_frame_equal(df, df_original)

    def test_select_columns_with_missing_column(self):
        """Test selecting a non-existent column raises KeyError"""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        with self.assertRaises(KeyError):
            PandasAdapter.select_columns(df, ["a", "missing"])

    def test_select_no_columns(self):
        """Test selecting an empty list of columns"""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

        result_df = PandasAdapter.select_columns(df, [])

        expected_df = pd.DataFrame(index=df.index)
        expected_df.columns = pd.Index([], dtype="object")
        pd.testing.assert_frame_equal(result_df.reset_index(drop=True), expected_df)


class TestPandasAdapterDropColumns:
    def test_drop_existing_columns(self):
        # Arrange
        df = pd.DataFrame(
            {
                "col1": ["A", "B"],
                "col2": ["X", "Y"],
                "col3": [1, 2],
            }
        )

        # Act
        result_df = PandasAdapter.drop_columns(df, ["col2"])

        # Assert
        assert "col2" not in result_df.columns
        assert list(result_df.columns) == ["col1", "col3"]

    def test_drop_missing_column_raises(self):
        # Arrange
        df = pd.DataFrame(
            {
                "col1": ["A"],
                "col2": ["B"],
            }
        )

        # Act & Assert
        with pytest.raises(MissingColumnError):
            PandasAdapter.drop_columns(df, ["does_not_exist"])


class TestPandasAdapterColumnHasNulls:
    def test_column_has_nulls_true(self):
        # Arrange
        df = pd.DataFrame(
            {
                "col1": ["A", "B"],
                "col2": [1, None],
            }
        )

        # Act
        has_nulls = PandasAdapter.column_has_nulls(df, "col2")

        # Assert
        assert has_nulls is True

    def test_column_has_nulls_false(self):
        # Arrange
        df = pd.DataFrame(
            {
                "col1": ["A", "B"],
                "col2": [1, 2],
            }
        )

        # Act
        has_nulls = PandasAdapter.column_has_nulls(df, "col2")

        # Assert
        assert has_nulls is False

    def test_column_has_nulls_missing_column_raises(self):
        # Arrange
        df = pd.DataFrame(
            {
                "col1": ["A"],
                "col2": ["B"],
            }
        )

        # Act & Assert
        with pytest.raises(MissingColumnError):
            PandasAdapter.column_has_nulls(df, "not_here")


class TestPandasAdapterGroupbyAggregate(unittest.TestCase):
    """Test PandasAdapter.groupby_aggregate method"""

    def test_groupby_single_aggregation(self):
        """Test grouping with a single aggregation per column"""
        df = pd.DataFrame(
            {
                "category": ["A", "A", "B", "B", "B"],
                "value": [10, 20, 30, 40, 50],
            }
        )

        result_df = PandasAdapter.groupby_aggregate(
            df, groupby_cols=["category"], aggregations={"value": "sum"}
        )

        expected_df = pd.DataFrame(
            {
                "category": ["A", "B"],
                "value_sum": [30, 120],
            }
        )

        pd.testing.assert_frame_equal(result_df, expected_df)

        # Index is reset
        self.assertEqual(result_df.index.tolist(), [0, 1])

    def test_groupby_multiple_aggregations(self):
        """Test grouping with multiple aggregations per column"""
        df = pd.DataFrame(
            {
                "category": ["A", "A", "B", "B", "B"],
                "value": [10, 20, 30, 40, 50],
                "score": [1, 2, 3, 4, 5],
            }
        )

        result_df = PandasAdapter.groupby_aggregate(
            df,
            groupby_cols=["category"],
            aggregations={"value": ["sum", "mean"], "score": ["max", "min"]},
        )

        # Flattened column names: value_sum, value_mean, score_max, score_min
        expected_df = pd.DataFrame(
            {
                "category": ["A", "B"],
                "value_sum": [30, 120],
                "value_mean": [15.0, 40.0],
                "score_max": [2, 5],
                "score_min": [1, 3],
            }
        )

        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_groupby_multiple_group_columns(self):
        """Test grouping by multiple columns"""
        df = pd.DataFrame(
            {
                "group1": ["X", "X", "Y", "Y", "Y"],
                "group2": ["A", "A", "A", "B", "B"],
                "value": [1, 2, 3, 4, 5],
            }
        )

        result_df = PandasAdapter.groupby_aggregate(
            df, groupby_cols=["group1", "group2"], aggregations={"value": "sum"}
        )

        expected_df = pd.DataFrame(
            {"group1": ["X", "Y", "Y"], "group2": ["A", "A", "B"], "value_sum": [3, 3, 9]}
        )

        pd.testing.assert_frame_equal(result_df, expected_df)


class TestPandasAdapterUnion(unittest.TestCase):
    """Test PandasAdapter.union method"""

    def test_union_basic(self):
        """Test union of two DataFrames with overlapping rows"""
        df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df2 = pd.DataFrame({"a": [2, 5], "b": [4, 6]})

        result_df = PandasAdapter.union(df1, df2)
        expected_df = pd.DataFrame({"a": [1, 2, 5], "b": [3, 4, 6]})

        pd.testing.assert_frame_equal(result_df.reset_index(drop=True), expected_df)

    def test_union_no_overlap(self):
        """Test union of two DataFrames with no overlapping rows"""
        df1 = pd.DataFrame({"x": [1, 2]})
        df2 = pd.DataFrame({"x": [3, 4]})

        result_df = PandasAdapter.union(df1, df2)
        expected_df = pd.DataFrame({"x": [1, 2, 3, 4]})

        pd.testing.assert_frame_equal(result_df.reset_index(drop=True), expected_df)

    def test_union_empty_dataframe(self):
        """Test union when one DataFrame is empty"""
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"a": []})

        result_df = PandasAdapter.union(df1, df2)
        expected_df = pd.DataFrame({"a": [1, 2]})

        pd.testing.assert_frame_equal(
            result_df.reset_index(drop=True), expected_df, check_dtype=False
        )


class TestPandasAdapterRenameColumns(unittest.TestCase):
    """Test PandasAdapter.rename_columns method"""

    def test_rename_columns_basic(self):
        """Test renaming a subset of columns"""
        df = pd.DataFrame({"old1": [1], "old2": [2], "keep": [3]})
        rename_map = {"old1": "new1", "old2": "new2"}

        result_df = PandasAdapter.rename_columns(df, rename_map)
        expected_df = pd.DataFrame({"new1": [1], "new2": [2], "keep": [3]})

        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_rename_columns_no_change(self):
        """Test renaming with an empty map does nothing"""
        df = pd.DataFrame({"a": [1], "b": [2]})
        rename_map = {}

        result_df = PandasAdapter.rename_columns(df, rename_map)
        pd.testing.assert_frame_equal(result_df, df)

    def test_rename_columns_partial_map(self):
        """Test renaming with a partial map"""
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        rename_map = {"b": "B"}

        result_df = PandasAdapter.rename_columns(df, rename_map)
        expected_df = pd.DataFrame({"a": [1], "B": [2], "c": [3]})

        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_rename_columns_copy(self):
        """Ensure the returned DataFrame is a copy, not a view"""
        df = pd.DataFrame({"x": [1]})
        result_df = PandasAdapter.rename_columns(df, {"x": "y"})

        # Changing the result should not affect original df
        result_df["y"] = [99]
        self.assertEqual(df["x"].iloc[0], 1)


class TestPandasAdapterMaterializeData(unittest.TestCase):
    """Test PandasAdapter materialize_data method"""

    @patch("pandas.read_parquet")
    @patch("pathlib.Path.mkdir")
    def test_materialize_data_basic(self, mock_mkdir, mock_read_parquet):
        """Test materialize_data saves and reads parquet correctly"""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        mock_path = Path(PROCESSOR_CACHE_PATH) / "test_data.parquet"

        mock_read_result = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        mock_read_parquet.return_value = mock_read_result

        with patch.object(df, "to_parquet") as mock_to_parquet:
            result = PandasAdapter.materialize_data(df, name="test_data")

            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
            mock_to_parquet.assert_called_once_with(mock_path, engine="pyarrow", index=False)
            mock_read_parquet.assert_called_once_with(mock_path, engine="pyarrow")
            pd.testing.assert_frame_equal(result, mock_read_result)

    @patch("pandas.read_parquet")
    @patch("pathlib.Path.mkdir")
    def test_materialize_data_with_timestamp(self, mock_mkdir, mock_read_parquet):
        """Test materialize_data with timestamp creates correct path"""
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        timestamp = "2023-01-01_12-00-00"
        mock_path = Path(PROCESSOR_CACHE_PATH) / timestamp / "test_data.parquet"

        mock_read_result = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        mock_read_parquet.return_value = mock_read_result

        with patch.object(df, "to_parquet") as mock_to_parquet:
            result = PandasAdapter.materialize_data(df, name="test_data", timestamp=timestamp)

            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
            mock_to_parquet.assert_called_once_with(mock_path, engine="pyarrow", index=False)
            mock_read_parquet.assert_called_once_with(mock_path, engine="pyarrow")
            pd.testing.assert_frame_equal(result, mock_read_result)


class TestPandasAdapterIntegration(unittest.TestCase):
    """Integration tests for PandasAdapter full workflow"""

    @patch("pandas.read_csv")
    def test_full_etl_workflow(self, mock_read_csv):
        """Test complete ETL workflow"""
        # Create realistic test data
        raw_data = pd.DataFrame(
            {
                "id": [1, 2, np.nan, 3, 1, 4],
                "value": [10, 20, 30, np.nan, 10, 50],
                "name": ["A", "B", "C", "D", "A", "E"],
            }
        )
        mock_read_csv.return_value = raw_data

        # Execute workflow
        df = PandasAdapter.read_csv(path="/data/input.csv", sep=",", header=0)
        df = PandasAdapter.dropna(df, subset=["id", "value"])
        df = PandasAdapter.drop_duplicates(df, subset=["id"])

        # Verify read_csv was called correctly
        mock_read_csv.assert_called_once_with("/data/input.csv", sep=",", header=0)

        # Verify the final result
        expected = pd.DataFrame(
            {"id": [1.0, 2.0, 4.0], "value": [10.0, 20.0, 50.0], "name": ["A", "B", "E"]},
            index=[0, 1, 5],
        )
        pd.testing.assert_frame_equal(df, expected)

        # Test writing the result
        tmpdir = Path(tempfile.mkdtemp())
        output_path = tmpdir / "data"
        with patch.object(df, "to_csv") as mock_to_csv:
            PandasAdapter.to_csv(df, path=output_path, output_sep=",", index=False)
            mock_to_csv.assert_called_once_with(tmpdir / "data.csv", sep=",", index=False)


if __name__ == "__main__":
    unittest.main()
