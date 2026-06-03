import json
from pathlib import Path
from typing import Any
from unittest.mock import mock_open, patch

import pytest
import torch

from hakken_ml_toolkit.ml_base_structures import (
    Fact,
    FactIndex,
    KnowledgeGraph,
    Mapping,
)
from hakken_ml_toolkit.ml_base_structures.common.exceptions import (
    InvalidTriplesDictKeyError,
    MappingNotFoundError,
    SplitNotInTriplesError,
    TripleNotFoundError,
)


@pytest.fixture
def sample_kg_data() -> dict[str, Any]:
    return {
        "facts_dict": {
            "all": torch.tensor([[0, 0, 1], [1, 1, 2]], dtype=torch.long),
            "train": torch.tensor([[0, 0, 1]], dtype=torch.long),
            "val": torch.tensor([[1, 1, 2]], dtype=torch.long),
        },
        "num_entities": 3,
        "num_relations": 2,
        "entity_mapping": Mapping(
            id_to_index={"e0": 0, "e1": 1, "e2": 2},
            index_to_id={0: "e0", 1: "e1", 2: "e2"},
        ),
        "relation_mapping": Mapping(id_to_index={"r0": 0, "r1": 1}, index_to_id={0: "r0", 1: "r1"}),
    }


@pytest.fixture
def sample_temporal_kg_data() -> dict[str, Any]:
    return {
        "facts_dict": {
            "all": torch.tensor([[0, 0, 1, 0], [1, 1, 2, 1]], dtype=torch.long),
            "train": torch.tensor([[0, 0, 1, 0]], dtype=torch.long),
            "val": torch.tensor([[1, 1, 2, 1]], dtype=torch.long),
        },
        "num_entities": 3,
        "num_relations": 2,
        "num_timestamps": 2,
        "entity_mapping": Mapping(
            id_to_index={"e0": 0, "e1": 1, "e2": 2},
            index_to_id={0: "e0", 1: "e1", 2: "e2"},
        ),
        "relation_mapping": Mapping(id_to_index={"r0": 0, "r1": 1}, index_to_id={0: "r0", 1: "r1"}),
        "timestamp_mapping": Mapping(
            id_to_index={"t0": 0, "t1": 1}, index_to_id={0: "t0", 1: "t1"}
        ),
    }


@pytest.fixture
def mock_path(tmp_path: Path) -> Path:
    return tmp_path / "test_kg"


def test_knowledge_graph_init(sample_kg_data: dict[str, Any]) -> None:
    kg = KnowledgeGraph(**sample_kg_data)
    assert kg.num_entities == 3
    assert kg.num_relations == 2
    assert len(kg.facts_dict) == 3
    assert "all" in kg.facts_dict
    assert "train" in kg.facts_dict
    assert "val" in kg.facts_dict


def test_temporal_knowledge_graph_init(sample_temporal_kg_data: dict[str, Any]) -> None:
    kg = KnowledgeGraph(**sample_temporal_kg_data)
    assert kg.num_entities == 3
    assert kg.num_relations == 2
    assert kg.num_timestamps == 2
    assert kg.is_temporal()
    assert kg.timestamp_mapping is not None


def test_invalid_column() -> None:
    with pytest.raises(TypeError):
        KnowledgeGraph(
            facts_dict={"train": torch.tensor([[0, 0], [1, 1]], dtype=torch.long)},
        )


def test_invalid_data_type() -> None:
    with pytest.raises(TypeError):
        KnowledgeGraph(
            facts_dict={
                "train": torch.tensor([[0.0, 0.0, 0.0], [1.1, 1.1, 1.1]], dtype=torch.float)
            }
        )


def test_negative_value_error() -> None:
    with pytest.raises(TypeError):
        KnowledgeGraph(
            facts_dict={"train": torch.tensor([[-0, -0, -0], [-1, -1, -1]], dtype=torch.long)}
        )


def test_invalid_dimension_error() -> None:
    with pytest.raises(TypeError):
        KnowledgeGraph(facts_dict={"train": torch.tensor([], dtype=torch.long)})


def test_split_exception(sample_kg_data: dict[str, Any]) -> None:
    with pytest.raises(SplitNotInTriplesError):
        kg = KnowledgeGraph(**sample_kg_data)
        fact_batch = torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.long)
        kg.remove_fact_batch(split="test", fact_batch=fact_batch)


def test_encode_facts_exception(sample_kg_data: dict[str, Any]) -> None:
    with pytest.raises(TripleNotFoundError):
        kg = KnowledgeGraph(**sample_kg_data)
        kg.encode_facts(
            triples_list=[("invalid_subject", "invalid_relation", "invalid_object")],
            on_missing="raise",
        )


def test_triples_key() -> None:
    facts = torch.tensor([[0, 0, 1], [1, 1, 2]], dtype=torch.long)
    with pytest.raises(InvalidTriplesDictKeyError):
        KnowledgeGraph(
            facts_dict={"invalid_key": facts},
        )


def test_decode_fact(sample_kg_data: dict[str, Any]) -> None:
    kg = KnowledgeGraph(**sample_kg_data)
    fact_index: FactIndex = (0, 0, 1)
    fact = kg.decode_fact(fact_index)
    assert fact[0] == "e0"  # subject
    assert fact[1] == "r0"  # relation
    assert fact[2] == "e1"  # object


def test_decode_facts(sample_kg_data: dict[str, Any]) -> None:
    kg = KnowledgeGraph(**sample_kg_data)
    fact_indexes: list[FactIndex] = [(0, 0, 1)]
    facts = kg.decode_facts(fact_indexes)
    assert len(facts) == 1
    assert facts[0][0] == "e0"  # subject
    assert facts[0][1] == "r0"  # relation
    assert facts[0][2] == "e1"  # object


def test_encode_facts(sample_kg_data: dict[str, Any]) -> None:
    kg = KnowledgeGraph(**sample_kg_data)
    facts: list[Fact] = [("e0", "r0", "e1")]
    fact_indexes = kg.encode_facts(facts, on_missing="raise")
    assert len(fact_indexes) == 1
    assert fact_indexes[0] == (0, 0, 1)


def test_encode_facts_as_tensor(sample_kg_data: dict[str, Any]) -> None:
    kg = KnowledgeGraph(**sample_kg_data)
    facts: list[Fact] = [("e0", "r0", "e1"), ("e1", "r1", "e2")]
    fact_tensor = kg.encode_facts_as_tensor(facts, on_missing="raise")
    assert fact_tensor.shape == (2, 3)
    assert torch.equal(fact_tensor, torch.tensor([[0, 0, 1], [1, 1, 2]], dtype=torch.long))


def test_to_device(sample_kg_data: dict[str, Any]) -> None:
    kg = KnowledgeGraph(**sample_kg_data)
    # Mock the to method for tensors
    with patch.object(torch.Tensor, "to", return_value=torch.Tensor()):
        kg.to_device("cuda")
    # Verify the tensor.to method was called for each batch


@patch("builtins.open", new_callable=mock_open)
@patch("json.dump")
@patch("torch.save")
def test_knowledge_graph_save(
    mock_torch_save: Any,
    mock_json_dump: Any,
    _mock_file: Any,
    mock_path: Path,
    sample_kg_data: dict[str, Any],
) -> None:
    kg = KnowledgeGraph(**sample_kg_data)
    kg.save(mock_path)
    assert mock_torch_save.call_count == 3  # all, train, val
    assert mock_json_dump.call_count == 1
    # Check if the correct data is being saved
    saved_dict = mock_json_dump.call_args[0][0]
    assert saved_dict["num_entities"] == 3
    assert saved_dict["num_relations"] == 2


@patch("torch.load")
@patch("hakken_ml_toolkit.ml_base_structures.mapping.Mapping.load")
@patch("pathlib.Path.exists", return_value=True)
def test_knowledge_graph_load(
    _mock_exists: Any,
    mock_mapping_load: Any,
    mock_torch_load: Any,
    tmp_path: Path,
) -> None:
    data_json_path = tmp_path / "data.json"
    with open(data_json_path, "w") as f:
        json.dump({"num_entities": 20, "num_relations": 3}, f)
    # Mock tensor return values
    mock_tensor = torch.tensor([[0, 0, 1]], dtype=torch.long)
    mock_torch_load.return_value = mock_tensor
    # Mock mapping return values - need to handle MappingNotFoundError for timestamp_mapping
    mock_mapping = Mapping(id_to_index={}, index_to_id={})

    def mapping_side_effect(path):
        if "timestamp" in str(path) or "domain" in str(path):
            raise MappingNotFoundError(tmp_path)
        return mock_mapping

    mock_mapping_load.side_effect = mapping_side_effect
    kg = KnowledgeGraph.load(tmp_path)
    assert isinstance(kg, KnowledgeGraph)
    assert kg.num_entities == 20  # Should match the JSON data
    assert kg.num_relations == 3  # Should match the JSON data
    # Verify file read operations
    assert mock_torch_load.call_count > 0
    assert mock_mapping_load.call_count > 0


@patch("torch.load")
@patch("hakken_ml_toolkit.ml_base_structures.mapping.Mapping.load")
@patch("pathlib.Path.exists", return_value=True)
def test_temporal_knowledge_graph_load(
    _mock_exists: Any,
    mock_mapping_load: Any,
    mock_torch_load: Any,
    tmp_path: Path,
) -> None:
    # Create data.json for temporal KG
    data_json_path = tmp_path / "data.json"
    with open(data_json_path, "w") as f:
        json.dump({"num_entities": 20, "num_relations": 3, "num_timestamps": 10}, f)
    # Mock tensor return values - temporal KG has 4 columns
    mock_tensor = torch.tensor([[0, 0, 1, 0]], dtype=torch.long)
    mock_torch_load.return_value = mock_tensor
    # Mock mapping return values - for temporal KG, all mappings should load successfully
    mock_mapping = Mapping(id_to_index={}, index_to_id={})

    def mapping_side_effect(path):
        # For temporal KG, don't raise exception for timestamp_mapping
        if "domain" in str(path):
            raise MappingNotFoundError(tmp_path)
        return mock_mapping

    mock_mapping_load.side_effect = mapping_side_effect
    kg = KnowledgeGraph.load(tmp_path)
    assert isinstance(kg, KnowledgeGraph)
    assert kg.num_entities == 20  # Should match JSON data
    assert kg.num_relations == 3  # Should match JSON data
    assert kg.num_timestamps == 10  # Should match JSON data
    assert kg.is_temporal()  # Should detect temporal nature from 4-column tensor
    # Check if all mappings were attempted to be loaded
    # entity_mapping, relation_mapping, timestamp_mapping, domain_mapping
    assert mock_mapping_load.call_count == 4
    # Verify timestamp mapping was successfully loaded (not None)
    assert kg.timestamp_mapping is not None


def test_entity_mapping():
    index_to_id = {0: "ocid0", 1: "ocid1"}
    id_to_index = {"ocid0": 0, "ocid1": 1}
    mapping = Mapping(id_to_index=id_to_index, index_to_id=index_to_id)
    assert mapping.id_to_index == id_to_index
    assert mapping.index_to_id == index_to_id


def test_relation_mapping():
    index_to_id = {0: "ocid0", 1: "ocid1"}
    id_to_index = {"ocid0": 0, "ocid1": 1}
    mapping = Mapping(index_to_id=index_to_id, id_to_index=id_to_index)
    assert mapping.index_to_id == index_to_id
    assert mapping.id_to_index == id_to_index


@patch("torch.load")
@patch("hakken_ml_toolkit.ml_base_structures.mapping.Mapping.load")
@patch("pathlib.Path.exists", return_value=True)
def test_knowledge_graph_load_with_domains(
    _mock_exists: Any,
    mock_mapping_load: Any,
    mock_torch_load: Any,
    tmp_path: Path,
) -> None:
    # Create data.json with domain information
    data_json_path = tmp_path / "data.json"
    with open(data_json_path, "w") as f:
        json.dump({"num_entities": 20, "num_relations": 3, "num_domains": 5}, f)
    # Create entity_to_domain.json
    entity_to_domain_path = tmp_path / "entity_to_domain.json"
    with open(entity_to_domain_path, "w") as f:
        json.dump({"0": 0, "1": 1, "2": 0, "3": 2, "4": 1}, f)
    # Mock tensor return values
    mock_tensor = torch.tensor([[0, 0, 1]], dtype=torch.long)
    mock_torch_load.return_value = mock_tensor
    # Mock mapping return values - all mappings load successfully for domain test
    mock_mapping = Mapping(id_to_index={}, index_to_id={})

    def mapping_side_effect(path):
        # Only timestamp_mapping raises exception (non-temporal KG)
        if "timestamp" in str(path):
            raise MappingNotFoundError(tmp_path)
        return mock_mapping

    mock_mapping_load.side_effect = mapping_side_effect
    kg = KnowledgeGraph.load(tmp_path)
    assert isinstance(kg, KnowledgeGraph)
    assert kg.num_entities == 20
    assert kg.num_relations == 3
    assert kg.num_domains == 5
    # Verify domain mapping was loaded
    assert kg.domain_mapping is not None
    # Verify entity_to_domain mapping was loaded and converted to int keys
    assert kg.entity_to_domain is not None
    assert kg.entity_to_domain == {0: 0, 1: 1, 2: 0, 3: 2, 4: 1}
    assert all(isinstance(k, int) for k in kg.entity_to_domain)
    # Check if all mappings were attempted to be loaded
    # entity_mapping, relation_mapping, timestamp_mapping, domain_mapping
    assert mock_mapping_load.call_count == 4
    # Verify timestamp mapping is None (non-temporal)
    assert kg.timestamp_mapping is None
