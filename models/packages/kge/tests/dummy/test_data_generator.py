import pytest
import torch
from hakken_ml_toolkit.ml_base_structures import KnowledgeGraph
from hakken_ml_toolkit.ml_base_structures.data_generator import DummyDataGenerator

from kge.common.validation import is_long_tensor_with_dim


@pytest.fixture
def dummy_data_generator():
    return DummyDataGenerator()


def test_so_batch(dummy_data_generator: DummyDataGenerator):
    batch_size = 10
    num_entities = 100
    device = "cpu"
    seed = 42

    so_batch = dummy_data_generator.so_batch(batch_size, num_entities, device, seed)

    assert is_long_tensor_with_dim(so_batch, dim=2)

    assert so_batch.data.shape == (batch_size, 2)
    assert so_batch.data.device.type == device
    assert torch.all(so_batch.data >= 0)
    assert torch.all(so_batch.data < num_entities)


def test_sro_batch(dummy_data_generator: DummyDataGenerator):
    batch_size = 10
    num_entities = 100
    num_relations = 50
    device = "cpu"
    seed = 42

    sro_batch = dummy_data_generator.sro_batch(
        batch_size, num_entities, num_relations, device, seed
    )

    sro_batch_data = sro_batch

    assert isinstance(sro_batch_data, torch.Tensor)
    assert sro_batch_data.shape == (batch_size, 3)
    assert sro_batch_data.device.type == device
    assert torch.all(sro_batch_data[:, [0, 2]] >= 0)
    assert torch.all(sro_batch_data[:, [0, 2]] < num_entities)
    assert torch.all(sro_batch_data[:, 1] >= 0)
    assert torch.all(sro_batch_data[:, 1] < num_relations)

    # Check for uniqueness
    unique_triples = set(map(tuple, sro_batch_data.tolist()))
    assert len(unique_triples) == batch_size


def test_entity_mapping(dummy_data_generator: DummyDataGenerator):
    num_entities = 100
    entity_mapping = dummy_data_generator.entity_mapping(num_entities)

    assert len(entity_mapping.id_to_index) == num_entities
    assert len(entity_mapping.index_to_id) == num_entities
    assert all(f"entity_{i}" in entity_mapping.id_to_index for i in range(num_entities))
    assert all(i in entity_mapping.index_to_id for i in range(num_entities))


def test_relation_mapping(dummy_data_generator: DummyDataGenerator):
    num_relations = 50
    relation_mapping = dummy_data_generator.relation_mapping(num_relations)

    assert len(relation_mapping.id_to_index) == num_relations
    assert len(relation_mapping.index_to_id) == num_relations
    assert all(f"relation_{i}" in relation_mapping.id_to_index for i in range(num_relations))
    assert all(i in relation_mapping.index_to_id for i in range(num_relations))


def test_knowledge_graph(dummy_data_generator: DummyDataGenerator):
    batch_size = 10
    num_entities = 100
    num_relations = 50
    device = "cpu"
    seed = 42

    kg = dummy_data_generator.knowledge_graph(batch_size, num_entities, num_relations, device, seed)

    assert isinstance(kg, KnowledgeGraph)
    assert kg.num_entities == num_entities
    assert kg.num_relations == num_relations
    assert "all" in kg.facts_dict
    assert kg.facts_dict["all"].data.shape == (batch_size, 3)
    assert len(kg.entity_mapping.id_to_index) == num_entities
    assert len(kg.relation_mapping.id_to_index) == num_relations

    kg_2 = dummy_data_generator.knowledge_graph(
        batch_size, num_entities, num_relations, device, seed
    )

    triples = kg_2.facts_dict["all"].data
    triples_2 = kg_2.facts_dict["all"].data
    assert torch.allclose(triples, triples_2)
