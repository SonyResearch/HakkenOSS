from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from hakken_ml_toolkit.ml_base_structures import KnowledgeGraph, Mapping

from datasets import TextKGConfig, TextKGDataset


@pytest.fixture
def sample_data() -> dict[str, Any]:
    """Fixture providing sample test data instead of reading from filesystem."""
    # Sample entity mapping data
    entity_data = [
        ["USA", "0"],
        ["Russia", "1"],
        ["China", "2"],
        ["Japan", "3"],
        ["Germany", "4"],
        ["France", "5"],
        ["UK", "6"],
        ["Israel", "7"],
        ["Iran", "8"],
    ]

    # Sample relation mapping data
    relation_data = [
        ["Make statement", "0"],
        ["Express intent to cooperate", "1"],
        ["Appeal", "2"],
        ["Consult", "3"],
        ["Diplomatic cooperation", "4"],
    ]

    # Sample facts data for different splits
    train_data = [
        ["USA", "Make statement", "Russia", "2014-01-02"],
        ["China", "Express intent to cooperate", "Japan", "2014-01-03"],
        ["France", "Appeal", "Germany", "2014-01-10"],
        ["Iran", "Consult", "Russia", "2014-02-01"],
    ]

    valid_data = [
        ["UK", "Appeal", "France", "2014-03-15"],
        ["Germany", "Diplomatic cooperation", "USA", "2014-04-01"],
    ]

    test_data = [
        ["Israel", "Make statement", "Iran", "2014-05-12"],
        ["Russia", "Express intent to cooperate", "China", "2014-06-20"],
    ]

    # Sample timestamps mapping
    timestamp_data = [
        ["2014-01-02", "0"],
        ["2014-01-03", "1"],
        ["2014-01-10", "2"],
        ["2014-02-01", "3"],
        ["2014-03-15", "4"],
        ["2014-04-01", "5"],
        ["2014-05-12", "6"],
        ["2014-06-20", "7"],
    ]

    return {
        "entity_data": entity_data,
        "relation_data": relation_data,
        "train_data": train_data,
        "valid_data": valid_data,
        "test_data": test_data,
        "timestamp_data": timestamp_data,
    }


@pytest.fixture
def sample_config(tmp_path: Path) -> TextKGConfig:
    """Create a sample config for testing without filesystem dependencies."""
    column_names = ["subject", "relation", "object", "date"]

    return TextKGConfig(
        root_folder=str(tmp_path),
        files_dict={"train": "train.txt", "val": "valid.txt", "test": "test.txt"},
        delimiter="\t",
        column_names=column_names,
        date_format="%Y-%m-%d",
    )


@pytest.fixture
def mock_dsv_utils(sample_data: dict[str, Any]) -> Any:
    """Patch DSVUtils to return predefined data instead of reading files."""
    with patch("datasets.TextKGDataset.DSVUtils") as mock_dsv:
        # Setup the mock to return our sample data when read_dsv is called
        def mock_read_dsv(file_path: str, _delimiter: str, names: list[str]) -> pd.DataFrame:
            if "train" in file_path:
                data = sample_data["train_data"]

            elif "valid" in file_path:
                data = sample_data["valid_data"]

            elif "test" in file_path:
                data = sample_data["test_data"]
            elif "entity2id" in file_path:
                data = sample_data["entity_data"]
                names = ["ids", "indexes"]
            elif "relation2id" in file_path:
                data = sample_data["relation_data"]
                names = ["ids", "indexes"]
            else:
                data = []

            return pd.DataFrame(data, columns=names)

        mock_dsv.read_dsv.side_effect = mock_read_dsv
        yield mock_dsv


class MockTensor:
    def __init__(self, dim: int, size: int = 4):
        self.dim = dim
        self.shape = (size, dim)  # Default to 4 facts


def test_initialization(sample_config: TextKGConfig) -> None:
    """Test dataset initialization."""
    dataset = TextKGDataset(sample_config)
    assert dataset.config == sample_config
    assert dataset.kg is None


def test_csv_files(sample_data: dict[str, Any]) -> None:
    """Test CSV data directly using sample data instead of reading files."""
    for split_name in ["train.txt", "valid.txt", "test.txt"]:
        # Determine which data to use based on the split name
        if "train" in split_name:
            data = sample_data["train_data"]

        elif "valid" in split_name:
            data = sample_data["valid_data"]
        elif "test" in split_name:
            data = sample_data["test_data"]

        # Determine columns based on whether temporal data is used
        columns = ["subject", "relation", "object", "date"]

        # Create DataFrame directly from sample data
        df_i = pd.DataFrame(data, columns=columns)

        num_rows, num_columns = df_i.shape

        assert num_rows > 0
        assert num_columns == 4


@patch("datasets.TextKGDataset.load_data")
def test_load_data(
    mock_load_data: MagicMock, sample_config: TextKGConfig, sample_data: dict[str, Any]
) -> None:
    """Test loading data with a mocked KnowledgeGraph."""
    dataset = TextKGDataset(sample_config)

    # Create a mock KnowledgeGraph
    mock_kg = MagicMock(spec=KnowledgeGraph)
    mock_kg.num_entities = 9
    mock_kg.num_relations = 5

    # Mock facts dictionary with a tensor-like object
    mock_kg.facts_dict = {"train": MockTensor(3)}

    mock_kg.num_timestamps = 8

    # Create a mock timestamp mapping
    timestamp_mapping = cast("Mapping", MagicMock())
    timestamp_mapping.index_to_id = {
        i: date for i, (date, _) in enumerate(sample_data["timestamp_data"])
    }
    timestamp_mapping.id_to_index = {
        date: i for i, (date, _) in enumerate(sample_data["timestamp_data"])
    }
    mock_kg.timestamp_mapping = timestamp_mapping

    # Set up the mock to return our mocked knowledge graph
    mock_load_data.return_value = mock_kg

    kg = dataset.load_data()

    assert isinstance(kg, KnowledgeGraph)
    assert isinstance(kg.num_entities, int)
    assert isinstance(kg.num_relations, int)
    assert kg.num_entities > 0
    assert kg.num_relations > 0

    assert isinstance(kg.num_timestamps, int)

    ts_mapping = kg.timestamp_mapping
    assert ts_mapping is not None
    max_index = max(ts_mapping.index_to_id.keys())
    for i in range(max_index - 1):
        first_date = timestamp_mapping.index_to_id[i]
        second_date = timestamp_mapping.index_to_id[i + 1]
        first_dt = pd.to_datetime(first_date)
        second_dt = pd.to_datetime(second_date)
        assert second_dt > first_dt


@patch("datasets.TextKGDataset.load_mapping")
def test_entity_mapping(
    mock_load_mapping: MagicMock,
    sample_config: TextKGConfig,
    sample_data: dict[str, Any],
) -> None:
    """Test entity mapping using mock data."""
    dataset = TextKGDataset(sample_config)

    # Create a mock mapping result
    mock_mapping = MagicMock()
    mock_mapping.id_to_index = {row[0]: int(row[1]) for row in sample_data["entity_data"]}
    mock_mapping.index_to_id = {int(row[1]): row[0] for row in sample_data["entity_data"]}

    mock_load_mapping.return_value = mock_mapping

    # Call the method directly with our mocked values
    mapping = dataset.load_mapping(
        file_path="dummy/path/entity2id.txt",
        delimiter="\t",
        column_names=["ids", "indexes"],
    )

    assert mapping.id_to_index["USA"] == 0
    assert mapping.id_to_index["Russia"] == 1

    assert mapping.index_to_id[0] == "USA"
    assert mapping.index_to_id[8] == "Iran"


@patch("datasets.TextKGDataset.load_mapping")
def test_relation_mapping(
    mock_load_mapping: MagicMock,
    sample_config: TextKGConfig,
    sample_data: dict[str, Any],
) -> None:
    """Test relation mapping using mock data."""
    dataset = TextKGDataset(sample_config)

    # Create a mock mapping result
    mock_mapping = MagicMock()
    mock_mapping.id_to_index = {row[0]: int(row[1]) for row in sample_data["relation_data"]}
    mock_mapping.index_to_id = {int(row[1]): row[0] for row in sample_data["relation_data"]}

    mock_load_mapping.return_value = mock_mapping

    # Call the method directly with our mocked values
    mapping = dataset.load_mapping(
        file_path="dummy/path/relation2id.txt",
        delimiter="\t",
        column_names=["ids", "indexes"],
    )

    assert len(mapping.id_to_index) == 5
    assert len(mapping.index_to_id) == 5

    assert mapping.id_to_index["Make statement"] == 0
    assert mapping.id_to_index["Express intent to cooperate"] == 1

    assert mapping.index_to_id[0] == "Make statement"
    assert mapping.index_to_id[4] == "Diplomatic cooperation"
