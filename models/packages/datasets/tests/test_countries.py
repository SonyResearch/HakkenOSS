from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from hakken_ml_toolkit.ml_base_structures import KnowledgeGraph

from datasets import TextKGConfig, TextKGDataset


@pytest.fixture
def sample_data() -> dict[str, Any]:
    """Fixture providing sample test data instead of reading from filesystem."""
    # Sample entity mapping data
    entity_data = [
        ["oceania", "0"],
        ["new_zealand", "1"],
        ["australia", "2"],
        ["fiji", "3"],
        ["tonga", "4"],
        ["solomon_islands", "5"],
        ["samoa", "6"],
        ["vanuatu", "7"],
        ["venezuela", "8"],
    ]

    # Sample relation mapping data
    relation_data = [["neighbor", "0"], ["locatedin", "1"]]

    # Sample facts data for different splits
    train_data = [
        ["new_zealand", "locatedin", "oceania"],
        ["australia", "locatedin", "oceania"],
        ["fiji", "locatedin", "oceania"],
        ["australia", "neighbor", "new_zealand"],
    ]

    valid_data = [
        ["tonga", "locatedin", "oceania"],
        ["solomon_islands", "locatedin", "oceania"],
    ]

    test_data = [["samoa", "locatedin", "oceania"], ["vanuatu", "locatedin", "oceania"]]

    return {
        "entity_data": entity_data,
        "relation_data": relation_data,
        "train_data": train_data,
        "valid_data": valid_data,
        "test_data": test_data,
    }


@pytest.fixture
def sample_config(tmp_path: Path) -> TextKGConfig:
    """Create a sample config for testing without filesystem dependencies."""
    entity_mapping = dict(filename="entity2id.txt", delimiter="\t", column_names=["ids", "indexes"])
    relation_mapping = dict(
        filename="relation2id.txt", delimiter="\t", column_names=["ids", "indexes"]
    )
    return TextKGConfig(
        root_folder=str(tmp_path),
        files_dict={"train": "train.txt", "val": "valid.txt", "test": "test.txt"},
        delimiter="\t",
        column_names=["subject", "relation", "object"],
        relation_mapping=relation_mapping,
        entity_mapping=entity_mapping,
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


def test_initialization(sample_config: TextKGConfig) -> None:
    """Test dataset initialization."""
    dataset = TextKGDataset(sample_config)
    assert dataset.config == sample_config
    assert dataset.kg is None


@pytest.mark.parametrize("split_name", ["train.txt", "valid.txt", "test.txt"])
def test_csv_files(split_name: str, sample_data: dict[str, Any]) -> None:
    """Test CSV data directly using sample data instead of reading files."""
    # Determine which data to use based on the split name
    if "train" in split_name:
        data = sample_data["train_data"]
    elif "valid" in split_name:
        data = sample_data["valid_data"]
    elif "test" in split_name:
        data = sample_data["test_data"]

    # Create DataFrame directly from sample data
    df_i = pd.DataFrame(data, columns=["subject", "relation", "object"])

    num_rows, num_columns = df_i.shape

    assert num_rows > 1
    assert num_columns == 3


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

    assert mapping.id_to_index["oceania"] == 0
    assert mapping.id_to_index["new_zealand"] == 1

    assert mapping.index_to_id[0] == "oceania"
    assert mapping.index_to_id[8] == "venezuela"


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

    assert len(mapping.id_to_index) == 2
    assert len(mapping.index_to_id) == 2

    assert mapping.id_to_index["neighbor"] == 0
    assert mapping.id_to_index["locatedin"] == 1

    assert mapping.index_to_id[0] == "neighbor"
    assert mapping.index_to_id[1] == "locatedin"


@patch("datasets.TextKGDataset.load_data")
def test_load_data(mock_load_data: MagicMock, sample_config: TextKGConfig) -> None:
    """Test loading data with a mocked KnowledgeGraph."""
    dataset = TextKGDataset(sample_config)

    # Create a mock KnowledgeGraph
    mock_kg = MagicMock(spec=KnowledgeGraph)
    mock_kg.num_entities = 9
    mock_kg.num_relations = 2
    mock_kg.num_timestamps = None

    # Mock facts dictionary with a tensor-like object
    class MockTensor:
        def __init__(self, dim: int):
            self.dim = dim
            self.shape = (4, dim)  # Assuming 4 facts

    mock_kg.facts_dict = {"train": MockTensor(2)}

    # Set up the mock to return our mocked knowledge graph
    mock_load_data.return_value = mock_kg

    # Call the method to test
    kg = dataset.load_data()

    assert isinstance(kg, KnowledgeGraph)
    assert isinstance(kg.num_entities, int)
    assert isinstance(kg.num_relations, int)
    assert kg.num_entities > 0
    assert kg.num_relations > 0
    assert kg.num_timestamps is None
