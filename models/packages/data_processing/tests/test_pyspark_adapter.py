import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from pyspark.sql import Row, SparkSession

from data_processing.adapters.impl.pyspark_adapter import PySparkAdapter
from data_processing.utils.errors import (
    ColumnCountMismatchError,
    EmptyListError,
    MissingColumnError,
    MissingPysparkBuilderError,
)
from data_processing.values import DEFAULT_SEPARATOR, PROCESSOR_CACHE_PATH


class PySparkTestCase(unittest.TestCase):
    """Base class for Spark tests with SparkSession setup/teardown"""

    @classmethod
    def setUpClass(cls):
        try:
            cls.spark = (
                SparkSession.builder.master("local[1]").appName("PySparkAdapterTest").getOrCreate()
            )
        except Exception as e:
            msg = f"Skipping PySparkAdapter tests: failed to init SparkSession: {e}"
            raise unittest.SkipTest(msg) from e

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()


class TestPySparkAdapterReadCSV(PySparkTestCase):
    """Test PySparkAdapter read_csv method"""

    @patch("pyspark.sql.SparkSession.read", new_callable=Mock)
    def test_read_csv_defaults(self, mock_read):
        """Test read_csv with default parameters"""
        # Create a real Spark DataFrame
        df_real = self.spark.createDataFrame([Row(col1=1, col2=3), Row(col1=2, col2=4)])
        mock_read.csv.return_value = df_real

        result = PySparkAdapter.read_csv(path="/local/data.csv", spark_builder=self.spark.builder)

        mock_read.csv.assert_called_once_with(
            "/local/data.csv",
            sep=DEFAULT_SEPARATOR,
            header=True,
            schema=None,
            inferSchema=True,
        )

        result_sorted = result.orderBy("col1").collect()
        expected_sorted = df_real.orderBy("col1").collect()
        self.assertEqual(result_sorted, expected_sorted)

    def test_missing_pyspark_builder(self):
        """Test read_csv without the pyspark builder"""
        # Create a real Spark DataFrame
        with self.assertRaises(MissingPysparkBuilderError):
            PySparkAdapter.read_csv(path="/local/data.csv")

    @patch("pyspark.sql.SparkSession.read", new_callable=Mock)
    def test_column_name_assignment(self, mock_read):
        """Test read_csv with the columns in correct amount"""
        # Create a real Spark DataFrame
        df_real = self.spark.createDataFrame([Row(col1=1, col2=3), Row(col1=2, col2=4)])
        mock_read.csv.return_value = df_real

        result = PySparkAdapter.read_csv(
            path="/local/data.csv",
            spark_builder=self.spark.builder,
            header=True,
            column_names=["hello", "morgen"],
        )

        assert result.columns == ["hello", "morgen"]
        assert result.count() == 2

    @patch("pyspark.sql.SparkSession.read", new_callable=Mock)
    def test_column_mismatch_error(self, mock_read):
        """Test read_csv with the columns in incorrect amount"""
        # Create a real Spark DataFrame
        df_real = self.spark.createDataFrame([Row(col1=1, col2=3), Row(col1=2, col2=4)])
        mock_read.csv.return_value = df_real

        with self.assertRaises(ColumnCountMismatchError):
            PySparkAdapter.read_csv(
                path="/local/data.csv",
                spark_builder=self.spark.builder,
                header=True,
                column_names=["col1", "col2", "col3"],
            )

        with self.assertRaises(ColumnCountMismatchError):
            PySparkAdapter.read_csv(
                path="/local/data.csv",
                spark_builder=self.spark.builder,
                header=None,
                column_names=["col1", "col2", "col3"],
            )


class TestPySparkAdapterConcat(PySparkTestCase):
    """Test PySparkAdapter concat method"""

    def test_concat_real_df(self):
        """Test concat correctly"""
        df_real = self.spark.createDataFrame([Row(a=1, b=2), Row(a=3, b=4)])
        df_result = PySparkAdapter.concat([df_real, df_real])

        data = [row.a for row in df_result.collect()]
        self.assertEqual(data, [1, 3, 1, 3])

        data = [row.b for row in df_result.collect()]
        self.assertEqual(data, [2, 4, 2, 4])

    def test_concat_empty_and_single_list(self):
        """Test concat with single element and empty list"""
        # Single element list
        df_real = self.spark.createDataFrame([Row(a=1, b=2), Row(a=3, b=4)])
        df_result = PySparkAdapter.concat([df_real])

        data = [row.a for row in df_result.collect()]
        self.assertEqual(data, [1, 3])

        data = [row.b for row in df_result.collect()]
        self.assertEqual(data, [2, 4])

        # Empty list
        with self.assertRaises(EmptyListError):
            PySparkAdapter.concat([])


class TestPySparkAdapterDropNA(PySparkTestCase):
    """Test PySparkAdapter dropna method"""

    def test_dropna_real_df(self):
        """Test dropna removes nulls correctly"""
        df_real = self.spark.createDataFrame([Row(a=1), Row(a=None), Row(a=3)])
        df_result = PySparkAdapter.dropna(df_real)
        data = [row.a for row in df_result.collect()]
        self.assertEqual(data, [1, 3])


class TestPySparkAdapterDropDuplicates(PySparkTestCase):
    """Test PySparkAdapter drop_duplicates method"""

    def test_drop_duplicates_real_df(self):
        """Test drop_duplicates removes duplicates correctly"""
        df_real = self.spark.createDataFrame([Row(a=1), Row(a=1), Row(a=2)])
        df_result = PySparkAdapter.drop_duplicates(df_real)
        data = [row.a for row in df_result.collect()]
        self.assertEqual(sorted(data), [1, 2])


class TestSparkAdapterSplitColumn(PySparkTestCase):
    def test_split_column(self):
        df = self.spark.createDataFrame(
            [("Alice|Smith",), ("Bob|Johnson",), ("Charlie|Brown",), ("Dana",)], ["full_name"]
        )

        result_df = PySparkAdapter.split_column(
            df, "full_name", "first_name", "last_name", separator="\\|"
        )
        result = [tuple(row) for row in result_df.select("first_name", "last_name").collect()]

        expected = [("Alice", "Smith"), ("Bob", "Johnson"), ("Charlie", "Brown"), ("Dana", None)]
        self.assertEqual(result, expected)


class TestSparkAdapterFilterByValue(PySparkTestCase):
    def test_filter_by_value(self):
        df = self.spark.createDataFrame(
            [("A", "X"), ("-", "Y"), ("C", "-"), ("D", "Z")], ["col1", "col2"]
        )
        result_df = PySparkAdapter.filter_by_value(df, ["col1", "col2"], invalid_value="-")
        result = [tuple(row) for row in result_df.collect()]

        expected = [("A", "X"), ("D", "Z")]
        self.assertEqual(result, expected)

    def test_filter_with_custom_invalid_value(self):
        df = self.spark.createDataFrame(
            [("OK", "OK"), ("INVALID", "OK"), ("OK", "INVALID")], ["col1", "col2"]
        )
        result_df = PySparkAdapter.filter_by_value(df, ["col1", "col2"], invalid_value="INVALID")
        result = [tuple(row) for row in result_df.collect()]
        expected = [("OK", "OK")]
        self.assertEqual(result, expected)


class TestSparkAdapterFilterByCondition(PySparkTestCase):
    def test_filter_by_simple_condition(self):
        df = self.spark.createDataFrame(
            [(25, "Alice"), (40, "Bob"), (18, "Charlie"), (30, "Dana")], ["age", "name"]
        )
        result_df = PySparkAdapter.filter_by_condition(df, "age >= 30")
        result = [tuple(row) for row in result_df.collect()]
        expected = [(40, "Bob"), (30, "Dana")]
        self.assertEqual(result, expected)


class TestSparkAdapterApplyFunctionToColumn(PySparkTestCase):
    def test_apply_function_str_return(self):
        df = self.spark.createDataFrame([("alice",), ("bob",), ("charlie",)], ["name"])

        def to_upper(x: str) -> str:
            return x.upper()

        result_df = PySparkAdapter.apply_function_to_column(df, "name", "upper_name", to_upper)
        result = [tuple(row) for row in result_df.select("name", "upper_name").collect()]
        expected = [("alice", "ALICE"), ("bob", "BOB"), ("charlie", "CHARLIE")]
        assert result == expected

    def test_apply_function_int_return(self):
        df = self.spark.createDataFrame([(1,), (2,), (3,)], ["value"])

        def double(x: int) -> int:
            return x * 2

        result_df = PySparkAdapter.apply_function_to_column(df, "value", "double_value", double)
        result = [tuple(row) for row in result_df.select("value", "double_value").collect()]
        expected = [(1, 2), (2, 4), (3, 6)]
        assert result == expected

    def test_apply_function_float_return(self):
        df = self.spark.createDataFrame([(1,), (2,), (3,)], ["value"])

        def half(x: int) -> float:
            return x / 2.0

        result_df = PySparkAdapter.apply_function_to_column(df, "value", "half_value", half)
        result = [tuple(row) for row in result_df.select("value", "half_value").collect()]
        expected = [(1, 0.5), (2, 1.0), (3, 1.5)]
        assert result == expected

    def test_apply_function_bool_return(self):
        df = self.spark.createDataFrame([(1,), (2,), (3,)], ["value"])

        def is_even(x: int) -> bool:
            return x % 2 == 0

        result_df = PySparkAdapter.apply_function_to_column(df, "value", "is_even", is_even)
        result = [tuple(row) for row in result_df.select("value", "is_even").collect()]
        expected = [(1, False), (2, True), (3, False)]
        assert result == expected

    def test_apply_function_default_type_when_no_hint(self):
        df = self.spark.createDataFrame([("a",), ("bb",), ("ccc",)], ["text"])

        # No type hints — should default to StringType
        def text_length(x):
            return len(x)

        result_df = PySparkAdapter.apply_function_to_column(df, "text", "length_str", text_length)
        result = [tuple(row) for row in result_df.select("text", "length_str").collect()]
        expected = [("a", "1"), ("bb", "2"), ("ccc", "3")]  # coerced to str
        assert result == expected


class TestSparkAdapterSelectColumns(PySparkTestCase):
    def test_select_columns_basic(self):
        df = self.spark.createDataFrame([(1, 4, 7), (2, 5, 8)], ["a", "b", "c"])
        result_df = PySparkAdapter.select_columns(df, ["a", "c"])
        result = [tuple(row) for row in result_df.collect()]
        expected = [(1, 7), (2, 8)]
        self.assertEqual(result, expected)


class TestSparkAdapterGroupbyAggregate(PySparkTestCase):
    def test_groupby_single_aggregation(self):
        df = self.spark.createDataFrame(
            [("A", 10), ("A", 20), ("B", 30), ("B", 40), ("B", 50)], ["category", "value"]
        )
        result_df = PySparkAdapter.groupby_aggregate(df, ["category"], {"value": "sum"})
        result = {row["category"]: row["value_sum"] for row in result_df.collect()}
        expected = {"A": 30, "B": 120}
        self.assertEqual(result, expected)

    def test_groupby_multiple_aggregations(self):
        """Test multiple aggregations per column"""
        df = self.spark.createDataFrame(
            [
                ("A", 10, 1),
                ("A", 20, 2),
                ("B", 30, 3),
                ("B", 40, 4),
                ("B", 50, 5),
            ],
            ["category", "value", "score"],
        )

        result_df = PySparkAdapter.groupby_aggregate(
            df,
            groupby_cols=["category"],
            aggregations={"value": ["sum", "mean"], "score": ["max", "min"]},
        )

        # Convert to dict of dicts for easy comparison
        result = {
            row["category"]: {
                "value_sum": row["value_sum"],
                "value_mean": row["value_mean"],
                "score_max": row["score_max"],
                "score_min": row["score_min"],
            }
            for row in result_df.collect()
        }

        expected = {
            "A": {"value_sum": 30, "value_mean": 15.0, "score_max": 2, "score_min": 1},
            "B": {"value_sum": 120, "value_mean": 40.0, "score_max": 5, "score_min": 3},
        }

        self.assertEqual(result, expected)

    def test_groupby_multiple_group_columns(self):
        """Test grouping by multiple columns"""
        df = self.spark.createDataFrame(
            [("X", "A", 1), ("X", "A", 2), ("Y", "A", 3), ("Y", "B", 4), ("Y", "B", 5)],
            ["group1", "group2", "value"],
        )

        result_df = PySparkAdapter.groupby_aggregate(
            df, groupby_cols=["group1", "group2"], aggregations={"value": "sum"}
        )

        # Convert to set for order-insensitive comparison
        result = {(row["group1"], row["group2"], row["value_sum"]) for row in result_df.collect()}
        expected = {("X", "A", 3), ("Y", "A", 3), ("Y", "B", 9)}

        self.assertEqual(result, expected)


class TestSparkAdapterUnion(PySparkTestCase):
    def test_union_basic(self):
        df1 = self.spark.createDataFrame([(1, 3), (2, 4)], ["a", "b"])
        df2 = self.spark.createDataFrame([(2, 4), (5, 6)], ["a", "b"])
        result_df = PySparkAdapter.union(df1, df2)
        result = {tuple(row) for row in result_df.collect()}
        expected = {(1, 3), (2, 4), (5, 6)}
        self.assertEqual(result, expected)


class TestSparkAdapterRenameColumns(PySparkTestCase):
    def test_rename_columns_basic(self):
        df = self.spark.createDataFrame([(1, 2, 3)], ["old1", "old2", "keep"])
        rename_map = {"old1": "new1", "old2": "new2"}
        result_df = PySparkAdapter.rename_columns(df, rename_map)
        self.assertEqual(result_df.columns, ["new1", "new2", "keep"])


class TestPySparkAdapterMaterializeData(PySparkTestCase):
    """Test PySparkAdapter materialize_data method"""

    @patch("pyspark.sql.DataFrame.write", new_callable=Mock)
    @patch("pyspark.sql.SparkSession.getActiveSession")
    def test_materialize_data_with_mocked_io(self, mock_get_active, mock_write):
        # Real Spark DataFrame
        df_real = self.spark.createDataFrame([Row(id=1, name="Alice"), Row(id=2, name="Bob")])

        # Mock SparkSession.read.parquet
        mock_spark_session = Mock()
        mock_spark_session.read.parquet.return_value = df_real
        mock_get_active.return_value = mock_spark_session

        # Mock the mode().parquet() chain
        mock_mode = Mock()
        mock_write.mode.return_value = mock_mode
        mock_mode.parquet.return_value = None

        timestamp = "20251003_210512"
        result_df = PySparkAdapter.materialize_data(df_real, "mydata", timestamp=timestamp)

        expected_path = Path(PROCESSOR_CACHE_PATH) / timestamp / "mydata"
        mock_mode.parquet.assert_called_once_with(str(expected_path))
        mock_spark_session.read.parquet.assert_called_once_with(str(expected_path))
        self.assertEqual(result_df.collect(), df_real.collect())


class TestSparkAdapterConcatenateColumns(PySparkTestCase):
    def test_concatenate_columns_basic(self):
        df = self.spark.createDataFrame([("A", "X"), ("B", "Y"), ("C", "Z")], ["col1", "col2"])
        result_df = PySparkAdapter.concatenate_columns(df, ["col1", "col2"], "combined")
        result = [tuple(row) for row in result_df.collect()]
        expected = [("A", "X", "A|X"), ("B", "Y", "B|Y"), ("C", "Z", "C|Z")]
        self.assertEqual(result, expected)

    def test_concatenate_columns_custom_delimiter(self):
        df = self.spark.createDataFrame([("A", "1"), ("B", "2")], ["first", "second"])
        result_df = PySparkAdapter.concatenate_columns(
            df, ["first", "second"], "joined", delimiter=","
        )
        result = [tuple(row) for row in result_df.collect()]
        expected = [("A", "1", "A,1"), ("B", "2", "B,2")]
        self.assertEqual(result, expected)

    def test_concatenate_columns_missing_column_raises(self):
        df = self.spark.createDataFrame([("A", "X")], ["col1", "col2"])
        with self.assertRaises(MissingColumnError):
            PySparkAdapter.concatenate_columns(df, ["col1", "col3"], "joined")


class TestSparkAdapterDropColumns(PySparkTestCase):
    def test_drop_existing_columns(self):
        # Arrange
        df = self.spark.createDataFrame([("A", "X", 1), ("B", "Y", 2)], ["col1", "col2", "col3"])

        # Act
        result_df = PySparkAdapter.drop_columns(df, ["col2"])

        # Assert
        assert "col2" not in result_df.columns
        assert "col1" in result_df.columns
        assert "col3" in result_df.columns

    def test_drop_missing_column_raises(self):
        # Arrange
        df = self.spark.createDataFrame([("A", "X")], ["col1", "col2"])

        # Act & Assert
        with self.assertRaises(MissingColumnError):
            PySparkAdapter.drop_columns(df, ["non_existent"])


class TestSparkAdapterColumnHasNulls(PySparkTestCase):
    def test_column_has_nulls_true(self):
        # Arrange
        df = self.spark.createDataFrame([("A", 1), ("B", None)], ["col1", "col2"])

        # Act
        has_nulls = PySparkAdapter.column_has_nulls(df, "col2")

        # Assert
        assert has_nulls is True

    def test_column_has_nulls_false(self):
        # Arrange
        df = self.spark.createDataFrame([("A", 1), ("B", 2)], ["col1", "col2"])

        # Act
        has_nulls = PySparkAdapter.column_has_nulls(df, "col2")

        # Assert
        assert has_nulls is False

    def test_column_has_nulls_missing_column_raises(self):
        # Arrange
        df = self.spark.createDataFrame([("A", "X")], ["col1", "col2"])

        # Act & Assert
        with self.assertRaises(MissingColumnError):
            PySparkAdapter.column_has_nulls(df, "not_here")
