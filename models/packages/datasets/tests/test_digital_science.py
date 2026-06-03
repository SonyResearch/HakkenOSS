import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from dotenv import load_dotenv
from hakken_ml_toolkit.ml_base_structures import KnowledgeGraph

from datasets import DigitalScience, DigitalScienceConfig
from datasets.common.domain import is_long_tensor_with_dim
from datasets.common.exceptions import (
    DataSplitProportionError,
    GraphNotLoadedError,
)

load_dotenv()


def sample_config(relation_filter: list[str] | None = None) -> DigitalScienceConfig:
    data_path = Path(__file__).parent / "data"

    return DigitalScienceConfig(
        root_folder=str(data_path),
        data_split_proportion_dict={"train": 0.5, "val": 0.5},
        relation_filter=relation_filter,
        nodes_file=str(data_path / "nodes.csv"),
        edges_file=str(data_path / "edges.csv"),
    )


def test_initialization():
    config = sample_config()

    digital_science = DigitalScience(config)
    assert digital_science.config == config
    assert digital_science.kg is None


def test_invalid_data_split_proportion() -> None:
    with pytest.raises(DataSplitProportionError):
        DigitalScience(
            DigitalScienceConfig(data_split_proportion_dict={"train": 0.8, "val": 0.1, "test": 0.2})
        )


def test_graph_not_loaded() -> None:
    num_r = None
    with pytest.raises(GraphNotLoadedError):
        data_repo = DigitalScience(
            DigitalScienceConfig(data_split_proportion_dict={"train": 0.5, "val": 0.5})
        )
        num_r = data_repo.num_relations

    assert num_r is None


@patch("hakken_ml_toolkit.ml_utils.DSVUtils.read_dsv")
@pytest.mark.parametrize("relation_filter", [None, ["R1", "R2"], ["R1"]])
def test_load_data(mock_read_csv, relation_filter: list[str] | None) -> None:
    config = sample_config(relation_filter)

    sample_df = pd.DataFrame(
        {
            "ocid_subject": ["A", "B", "C", "A"],
            "relation": ["R1", "R2", "R1", "R2"],
            "ocid_object": ["B", "C", "A", "C"],
        }
    )
    mock_read_csv.return_value = sample_df
    digital_science = DigitalScience(config)
    kg = digital_science.load_data()

    assert isinstance(kg, KnowledgeGraph)
    assert is_long_tensor_with_dim(kg.facts_dict["train"], dim=2)
    if relation_filter is not None and len(relation_filter) == 1:
        assert tuple(kg.facts_dict["train"].shape) == (1, 3)
        assert tuple(kg.facts_dict["val"].shape) == (1, 3)

        assert set(kg.entity_mapping.id_to_index.keys()) == {"A", "B", "C"}

        assert kg.num_entities == 3
        assert kg.num_relations == 1
    else:
        assert tuple(kg.facts_dict["train"].shape) == (2, 3)
        assert tuple(kg.facts_dict["val"].shape) == (2, 3)

        assert set(kg.entity_mapping.id_to_index.keys()) == {"A", "B", "C"}

        assert kg.num_entities == 3
        assert kg.num_relations == 2
    mock_read_csv.assert_called_once()


def test_save_config(tmp_path: Path) -> None:
    """Test that config is saved correctly to file."""
    config = sample_config()

    digital_science = DigitalScience(config)
    config_path = tmp_path / "data_repo/config.json"
    digital_science.save_config(config_path)

    assert config_path.exists()

    with open(str(config_path)) as f:
        saved_content_str = json.load(f)
        saved_content = json.loads(saved_content_str)

    config_dict = config.model_dump()
    for key, value in config_dict.items():
        print(f"{key}: {value} {type(saved_content)}")
        assert value == saved_content[key]


def test_load_config(tmp_path: Path) -> None:
    """Test that config is loaded correctly from file."""
    config = sample_config()

    config_path = tmp_path / "data_repo/config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config.model_dump_json(), f)

    loaded_config = DigitalScience.load_config(config_path)

    for key, value in config.model_dump().items():
        assert value == getattr(loaded_config, key)


def test_save(tmp_path: Path) -> None:
    # Arrange
    config = sample_config()
    digital_science = DigitalScience(config)
    save_path = tmp_path / "test_save"

    # Act
    digital_science.save(save_path)

    # Assert
    config_path = save_path / "config.json"
    assert config_path.exists(), "Config file should be created"
    assert config_path.is_file(), "Config file should be a file"


def test_load(tmp_path: Path) -> None:
    # Arrange
    config = sample_config()
    digital_science = DigitalScience(config)
    save_path = tmp_path / "test_load"
    digital_science.save(save_path)

    # Act
    loaded_ds = DigitalScience.load(save_path)

    # Assert
    assert loaded_ds is not None, "Loaded DigitalScience should not be None"
    assert isinstance(loaded_ds, DigitalScience), "Loaded object should be of type DigitalScience"
    assert loaded_ds.config.md5_hash() == config.md5_hash(), (
        "Loaded config should match original config"
    )


def test_hash_equality():
    """Test that identical configs have the same hash."""

    config1 = DigitalScienceConfig(nodes_file="path/to/nodes.csv", edges_file="path/to/edges.csv")
    config2 = DigitalScienceConfig(nodes_file="path/to/nodes.csv", edges_file="path/to/edges.csv")
    assert hash(config1) == hash(config2)


def test_hash_inequality():
    """Test that different configs have different hashes."""
    config1 = DigitalScienceConfig(nodes_file="path/to/nodes.csv", edges_file="path/to/edges.csv")
    config2 = DigitalScienceConfig(
        nodes_file="path/to/different_nodes.csv", edges_file="path/to/edges.csv"
    )
    assert hash(config1) != hash(config2)


def test_hash_with_path_objects():
    """Test that Path objects are handled correctly."""
    config1 = DigitalScienceConfig(
        nodes_file=Path("path/to/nodes.csv"), edges_file=Path("path/to/edges.csv")
    )
    config2 = DigitalScienceConfig(nodes_file="path/to/nodes.csv", edges_file="path/to/edges.csv")
    assert hash(config1) == hash(config2)


def test_hash_with_lists():
    """Test that lists are handled correctly."""
    config1 = DigitalScienceConfig(
        nodes_file="path/to/nodes.csv",
        edges_file="path/to/edges.csv",
        relation_filter=["rel1", "rel2", "rel3"],
    )
    config2 = DigitalScienceConfig(
        nodes_file="path/to/nodes.csv",
        edges_file="path/to/edges.csv",
        relation_filter=["rel1", "rel2", "rel3"],
    )
    assert hash(config1) == hash(config2)


def test_object_in_dict():
    """Test using the model as a dictionary key."""

    # Using dummy paths for testing purposes
    config1 = DigitalScienceConfig(nodes_file="path/to/nodes.csv", edges_file="path/to/edges.csv")
    config2 = DigitalScienceConfig(nodes_file="path/to/nodes.csv", edges_file="path/to/edges.csv")
    config3 = DigitalScienceConfig(nodes_file="different/path.csv", edges_file="path/to/edges.csv")

    # Create a dictionary with configs as keys
    data = {}
    data[config1] = "data for config1"

    # Should be able to retrieve with an equal config
    assert data.get(config2) == "data for config1"

    # Should not find a different config
    assert data.get(config3) is None
