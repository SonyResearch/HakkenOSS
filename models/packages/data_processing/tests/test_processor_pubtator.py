import json
from unittest.mock import mock_open, patch

import pandas as pd
import pytest

from data_processing.adapters.impl.pandas_adapter import PandasAdapter
from data_processing.data_processor.impl.processor_pubtator import PubtatorProcessor
from data_processing.utils.hashing import hash_string
from data_processing.values import (
    DOMAIN_PIPE_OBJECT_ID_COLUMN,
    DOMAIN_PIPE_SUBJECT_ID_COLUMN,
    NUMBER_OF_OCCURRENCES_COLUMN,
    OBJECT_DOMAIN_COLUMN,
    OBJECT_ID_COLUMN,
    OBJECT_ID_RAW_COLUMN,
    PMID_COLUMN,
    PMIDS_COLUMN,
    RELATION_ID_COLUMN,
    RELATION_TYPE_COLUMN,
    SUBJECT_DOMAIN_COLUMN,
    SUBJECT_ID_COLUMN,
    SUBJECT_ID_RAW_COLUMN,
    TIMESTAMP_COLUMN,
    YEAR_OCCURRENCES_COLUMN,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dummy_config(tmp_path):
    """Dummy configuration mimicking a Hydra config object."""

    class DummyFile:
        def __init__(self, path):
            self.path = path
            self.sep = "\t"
            self.column_names = [
                PMID_COLUMN,
                RELATION_TYPE_COLUMN,
                DOMAIN_PIPE_SUBJECT_ID_COLUMN,
                DOMAIN_PIPE_OBJECT_ID_COLUMN,
            ]
            self.header = False
            self.encoding = "utf-8"

    class DummyCfg:
        def __init__(self):
            self.data_files = type(
                "data_files", (), {"relations": [DummyFile(tmp_path / "test.tsv")]}
            )
            self.library = "pandas"
            self.spark_builder = None

        def build_spark(self):
            pass

    return DummyCfg()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_clean_removes_invalid_rows(dummy_config):
    """Test that _clean removes rows with '-' and NaN values correctly."""
    df = pd.DataFrame(
        {
            PMID_COLUMN: [1, 2, 3, 3],
            RELATION_TYPE_COLUMN: ["treat", "-", "associate", "relates_\to"],
            DOMAIN_PIPE_SUBJECT_ID_COLUMN: ["Chemical|C1", "-", "Disease|D2", "Chemical|D3"],
            DOMAIN_PIPE_OBJECT_ID_COLUMN: ["Gene|G1", "Gene|G2", None, "Gene|G3"],
            "other_column": [1, 2, 3, None],
        }
    )

    with patch(
        "data_processing.data_processor.processor_base.LibraryAdapterFactory.get_adapter",
        return_value=PandasAdapter(),
    ):
        processor = PubtatorProcessor(config=dummy_config)
        processor.df = df.copy()
        processor._clean()
        cleaned_df = processor.df

    assert len(cleaned_df) == 2
    assert cleaned_df.iloc[0][PMID_COLUMN] == 1
    assert cleaned_df.iloc[1][PMID_COLUMN] == 3
    assert SUBJECT_DOMAIN_COLUMN in cleaned_df.columns
    assert OBJECT_DOMAIN_COLUMN in cleaned_df.columns
    assert SUBJECT_ID_RAW_COLUMN in cleaned_df.columns
    assert OBJECT_ID_RAW_COLUMN in cleaned_df.columns


def test_generate_entity_id_adds_hash_columns(dummy_config):
    """Test that _generate_entity_id adds hashed subject/object IDs."""
    df = pd.DataFrame(
        {
            DOMAIN_PIPE_SUBJECT_ID_COLUMN: ["Chemical|C1", "Disease|D2"],
            DOMAIN_PIPE_OBJECT_ID_COLUMN: ["Gene|G1", "Gene|G2"],
        }
    )

    with patch(
        "data_processing.data_processor.processor_base.LibraryAdapterFactory.get_adapter",
        return_value=PandasAdapter(),
    ):
        processor = PubtatorProcessor(config=dummy_config)
        processor.df = df.copy()
        processor._generate_entity_id()
        result_df = processor.df

    assert SUBJECT_ID_COLUMN in result_df.columns
    assert OBJECT_ID_COLUMN in result_df.columns
    expected_hash = hash_string("Chemical|C1")
    assert result_df.iloc[0][SUBJECT_ID_COLUMN] == expected_hash
    assert isinstance(result_df.iloc[0][SUBJECT_ID_COLUMN], str)


def test_check_for_collisions_no_conflicts(dummy_config):
    """Test that _check_for_collisions passes when all hashed IDs are unique."""

    df = pd.DataFrame(
        {
            SUBJECT_ID_COLUMN: ["h1", "h2"],
            DOMAIN_PIPE_SUBJECT_ID_COLUMN: ["Chemical|C1", "Disease|D2"],
            OBJECT_ID_COLUMN: ["h3", "h4"],
            DOMAIN_PIPE_OBJECT_ID_COLUMN: ["Gene|G1", "Gene|G2"],
        }
    )

    with patch(
        "data_processing.data_processor.processor_base.LibraryAdapterFactory.get_adapter",
        return_value=PandasAdapter(),
    ):
        processor = PubtatorProcessor(config=dummy_config)
        processor.df = df.copy()

        # Should not raise, because all hashed IDs are unique
        processor._check_for_collisions()


def test_check_for_collisions_detects_collision(dummy_config):
    """Test that _check_for_collisions raises ValueError when collisions are found."""

    df = pd.DataFrame(
        {
            SUBJECT_ID_COLUMN: ["h1", "h1"],  # duplicate hash ID
            DOMAIN_PIPE_SUBJECT_ID_COLUMN: ["Chemical|C1", "Chemical|C2"],
            OBJECT_ID_COLUMN: ["h3", "h4"],
            DOMAIN_PIPE_OBJECT_ID_COLUMN: ["Gene|G1", "Gene|G2"],
        }
    )

    with patch(
        "data_processing.data_processor.processor_base.LibraryAdapterFactory.get_adapter",
        return_value=PandasAdapter(),
    ):
        processor = PubtatorProcessor(config=dummy_config)
        processor.df = df.copy()

        with pytest.raises(ValueError):
            processor._check_for_collisions()


def test_format_columns(dummy_config):
    """Test that _check_for_collisions raises ValueError when collisions are found."""

    df = pd.DataFrame(
        {
            SUBJECT_ID_COLUMN: ["h1", "h1"],  # duplicate hash ID
            SUBJECT_DOMAIN_COLUMN: ["Chemical", "CHEMICAL"],
            RELATION_TYPE_COLUMN: ["relates_to", "Treats"],
            OBJECT_ID_COLUMN: ["h3", "h4"],
            OBJECT_DOMAIN_COLUMN: ["Gene", "GENE"],
        }
    )

    with patch(
        "data_processing.data_processor.processor_base.LibraryAdapterFactory.get_adapter",
        return_value=PandasAdapter(),
    ):
        processor = PubtatorProcessor(config=dummy_config)
        processor.df = df.copy()

        processor._format_columns()

        # Validate that only the relevant columns were uppercased
        formatted_df = processor.df

        # Check uppercased columns
        assert all(formatted_df[OBJECT_DOMAIN_COLUMN] == ["GENE", "GENE"])
        assert all(formatted_df[SUBJECT_DOMAIN_COLUMN] == ["CHEMICAL", "CHEMICAL"])
        assert all(formatted_df[RELATION_TYPE_COLUMN] == ["RELATES_TO", "TREATS"])

        # Ensure unaffected columns remain unchanged
        assert all(formatted_df[SUBJECT_ID_COLUMN] == ["h1", "h1"])
        assert all(formatted_df[OBJECT_ID_COLUMN] == ["h3", "h4"])


def test_generate_relation_id(dummy_config):
    """Test that _generate_relation_id creates correct hashed relation IDs."""

    df = pd.DataFrame(
        {
            SUBJECT_DOMAIN_COLUMN: ["CHEMICAL", "GENE"],
            RELATION_TYPE_COLUMN: ["TREATS", "INTERACTS_WITH"],
            OBJECT_DOMAIN_COLUMN: ["DISEASE", "PROTEIN"],
        }
    )

    with patch(
        "data_processing.data_processor.processor_base.LibraryAdapterFactory.get_adapter",
        return_value=PandasAdapter(),
    ):
        processor = PubtatorProcessor(config=dummy_config)
        processor.df = df.copy()

        # Run the method
        processor._generate_relation_id()

        result_df = processor.df

        # Check that temporary column was created and contains concatenated values
        tmp_col = f"{RELATION_ID_COLUMN}_tmp"
        expected_concat = [
            "CHEMICAL\\|TREATS\\|DISEASE",
            "GENE\\|INTERACTS_WITH\\|PROTEIN",
        ]
        assert all(result_df[tmp_col] == expected_concat)

        # Check that RELATION_ID_COLUMN exists and is a hash of the concatenated string
        for _, row in result_df.iterrows():
            expected_hash = hash_string(row[tmp_col])
            assert row[RELATION_ID_COLUMN] == expected_hash

        # Ensure no NaNs and correct number of rows
        assert not result_df[RELATION_ID_COLUMN].isna().any()
        assert len(result_df) == len(df)


def test_get_unique_pmids(dummy_config):
    """Test that _get_unique_pmids selects and deduplicates PMIDs correctly."""
    df = pd.DataFrame(
        {
            "pmid": ["123", "456", "123", "789"],
            "title": ["A", "B", "C", "D"],
        }
    )

    with patch(
        "data_processing.data_processor.processor_base.LibraryAdapterFactory.get_adapter",
        return_value=PandasAdapter(),
    ):
        processor = PubtatorProcessor(config=dummy_config)
        processor.df = df.copy()

        # Spy on adapter methods
        result = processor._get_unique_pmids()

        assert set(result) == {"123", "456", "789"}
        assert len(result) == 3


def test_get_publication_year_adds_year_column(tmp_path, dummy_config):
    """Test that _get_publication_year correctly maps PMIDs to years and drops missing ones."""
    # --- Mock PubTator cache file content ---
    mock_jsonl = "\n".join(
        [
            json.dumps({"pmid": "1", "year": "2013"}),
            json.dumps({"pmid": "2", "year": "2015"}),
            json.dumps({"pmid": "3", "year": "2017"}),
            json.dumps({"pmid": "5", "year": None}),
        ]
    )

    # --- Create test dataframe ---
    df = pd.DataFrame(
        {
            PMID_COLUMN: [1, 2, 3, 4, 5],  # PMID 4 and 5 should be dropped
            "some_col": ["a", "b", "c", "d", "e"],
        }
    )

    with (
        patch(
            "data_processing.data_processor.processor_base.LibraryAdapterFactory.get_adapter",
            return_value=PandasAdapter(),
        ),
        patch(
            "builtins.open",
            mock_open(read_data=mock_jsonl),
        ),
        patch(
            "data_processing.data_processor.impl.processor_pubtator.PUBTATOR_PUBLICATION_METADATA_CACHE_FILE",
            tmp_path / "pubtator_cache.jsonl",
        ),
    ):
        processor = PubtatorProcessor(config=dummy_config)
        processor.df = df.copy()

        # Run the method under test
        processor._get_publication_year()

        result_df = processor.df

    # --- Assertions ---
    # Should only keep rows with PMIDs 1,2,3
    assert set(result_df[PMID_COLUMN]) == {1, 2, 3}
    assert len(result_df) == 3

    # Ensure TIMESTAMP_COLUMN was added and correctly mapped
    assert TIMESTAMP_COLUMN in result_df.columns
    expected_years = {"1": 2013, "2": 2015, "3": 2017}
    for _, row in result_df.iterrows():
        assert row[TIMESTAMP_COLUMN] == expected_years[str(row[PMID_COLUMN])]

    # No missing years remain
    assert not result_df[TIMESTAMP_COLUMN].isna().any()


def test_aggregate_triples_across_years_basic(dummy_config):
    """Test that _aggregate_triples_across_years aggregates and transforms correctly."""

    df = pd.DataFrame(
        {
            SUBJECT_ID_COLUMN: ["S1", "S1", "S1"],
            RELATION_TYPE_COLUMN: ["binds", "binds", "binds"],
            OBJECT_ID_COLUMN: ["O1", "O1", "O1"],
            TIMESTAMP_COLUMN: [2013, 2015, 2013],
            PMID_COLUMN: ["10", "20", "30"],
            SUBJECT_DOMAIN_COLUMN: ["gene", "gene", "gene"],
            OBJECT_DOMAIN_COLUMN: ["protein", "protein", "protein"],
            SUBJECT_ID_RAW_COLUMN: ["s1r", "s1r", "s1r"],
            OBJECT_ID_RAW_COLUMN: ["o1r", "o1r", "o1r"],
            RELATION_ID_COLUMN: ["r1", "r1", "r1"],
        }
    )

    with (
        patch(
            "data_processing.data_processor.processor_base.LibraryAdapterFactory.get_adapter",
            return_value=PandasAdapter(),
        ),
    ):
        processor = PubtatorProcessor(config=dummy_config)
        processor.df = df.copy()

        # Run
        processor._aggregate_triples_across_years()

        result_df = processor.df

    # --- Assertions ---
    # Single aggregated row
    assert len(result_df) == 1

    # Required columns exist
    expected_cols = {
        SUBJECT_ID_COLUMN,
        OBJECT_ID_COLUMN,
        RELATION_TYPE_COLUMN,
        TIMESTAMP_COLUMN,
        PMIDS_COLUMN,
        NUMBER_OF_OCCURRENCES_COLUMN,
        YEAR_OCCURRENCES_COLUMN,
    }
    assert expected_cols.issubset(result_df.columns)

    row = result_df.iloc[0]

    # Minimum year retained
    assert row[TIMESTAMP_COLUMN] == 2013

    # PMIDs concatenated
    assert row[PMIDS_COLUMN] == "10|20|30"

    # Unique sorted years concatenated
    assert row[YEAR_OCCURRENCES_COLUMN] == "2013|2015"

    # Occurrence count = 3
    assert row[NUMBER_OF_OCCURRENCES_COLUMN] == 3


def test_full_process_pipeline(dummy_config, tmp_path):
    """Test end-to-end process() with real PandasAdapter logic."""
    df = pd.DataFrame(
        {
            PMID_COLUMN: [1, 2],
            RELATION_TYPE_COLUMN: ["treat", "associate"],
            DOMAIN_PIPE_SUBJECT_ID_COLUMN: ["Chemical|C123", "Disease|D456"],
            DOMAIN_PIPE_OBJECT_ID_COLUMN: ["Gene|G789", "Gene|G111"],
        }
    )

    # Prepare fake pubtator cache file
    pubtator_cache = tmp_path / "pubtator_cache.jsonl"
    data = [
        {"pmid": "1", "year": 2020},
        {"pmid": "2", "year": 2021},
    ]
    with open(pubtator_cache, "w", encoding="utf-8") as f:
        for obj in data:
            f.write(json.dumps(obj) + "\n")

    with (
        patch(
            "data_processing.data_processor.processor_base.LibraryAdapterFactory.get_adapter",
            return_value=PandasAdapter(),
        ),
        patch(
            "data_processing.data_processor.impl.processor_pubtator.PUBTATOR_PUBLICATION_METADATA_CACHE_FILE",
            str(pubtator_cache),
        ),
        patch(
            "data_processing.data_processor.impl.processor_pubtator.fetch_pubtator_data",
            return_value=None,
        ) as mock_fetch,
        patch.object(PandasAdapter, "materialize_data", side_effect=lambda df, *_: df),
        patch.object(PandasAdapter, "count_rows", return_value=0),
        patch.object(PandasAdapter, "show_head", return_value=None),
    ):
        processor = PubtatorProcessor(config=dummy_config)
        # Mock load_data to return our dataframe
        processor.load_data = lambda _cfg: df.copy()

        result = processor.process()

    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(df)

    assert SUBJECT_DOMAIN_COLUMN in result.columns
    assert OBJECT_DOMAIN_COLUMN in result.columns
    assert SUBJECT_ID_COLUMN in result.columns
    assert OBJECT_ID_COLUMN in result.columns

    # Relation ID should be generated
    assert RELATION_ID_COLUMN in result.columns
    assert result[RELATION_ID_COLUMN].notna().all()

    # Year present and not null
    assert TIMESTAMP_COLUMN in result.columns
    assert result[TIMESTAMP_COLUMN].notna().all()
    assert set(result[TIMESTAMP_COLUMN]) == {2020, 2021}

    # Triples Aggregation results
    assert NUMBER_OF_OCCURRENCES_COLUMN in result.columns
    assert YEAR_OCCURRENCES_COLUMN in result.columns
    assert PMIDS_COLUMN in result.columns

    # Check the content
    assert result[NUMBER_OF_OCCURRENCES_COLUMN].notna().all()
    assert result[YEAR_OCCURRENCES_COLUMN].notna().all()
    assert result[PMIDS_COLUMN].notna().all()

    # PubTator fetch should have been called once with PMIDs [1, 2]
    mock_fetch.assert_called_once()
    called_pmids = mock_fetch.call_args[0][0]
    assert set(map(str, called_pmids)) == {"1", "2"}
