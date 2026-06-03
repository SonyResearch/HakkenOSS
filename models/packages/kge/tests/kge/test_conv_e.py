from pathlib import Path

import pytest
import torch

from kge.common.entities import KGEForwardOutput
from kge.common.validation import is_float_tensor_with_dim
from kge.models.conv_e import ConvE, ConvEConfig, ConvEModule


@pytest.fixture
def conve_config():
    return ConvEConfig(embedding_dim=100, num_entities=1000, num_relations=50)


@pytest.fixture
def conve_model(conve_config):
    return ConvE(conve_config)


def test_init(conve_model):
    assert isinstance(conve_model, ConvE)
    assert isinstance(conve_model.model, ConvEModule)


def test_compute_embedding_dimensions():
    assert ConvE.compute_embedding_dimensions(100) == (10, 10)
    assert ConvE.compute_embedding_dimensions(200) == (10, 20)

    with pytest.raises(Exception, match="Embedding dimension must be greater than 10"):
        ConvE.compute_embedding_dimensions(9)


def test_forward(conve_model: ConvE):
    subjects = torch.randint(0, 1000, (32,))
    relations = torch.randint(0, 50, (32,))
    objects = torch.randint(0, 1000, (32,))
    sro_batch = torch.stack([subjects, relations, objects], dim=1)
    output = conve_model.forward(sro_batch)

    assert isinstance(output, KGEForwardOutput)


def test_score(conve_model: ConvE):
    subjects = torch.randint(0, 1000, (32,))
    relations = torch.randint(0, 50, (32,))
    objects = torch.randint(0, 1000, (32,))
    sro_batch = torch.stack([subjects, relations, objects], dim=1)
    scores = conve_model.score(sro_batch)

    assert is_float_tensor_with_dim(scores, dim=2)
    assert scores.shape == (32, 1)


def test_entity_embeddings(conve_model: ConvE):
    # Create a batch of entity indices
    entity_batch = torch.tensor([0, 1, 2])

    # Get the embeddings
    embeddings = conve_model.entity_embeddings(entity_batch)

    # Check if the output is of the correct type
    assert is_float_tensor_with_dim(embeddings, dim=2)

    # Check if the shape is correct
    expected_shape = (3, conve_model.config.embedding_dim)
    assert embeddings.shape == expected_shape

    # Check if the output is a FloatTensor
    assert isinstance(embeddings, torch.FloatTensor)


def test_relation_embeddings(conve_model: ConvE):
    # Create a batch of entity indices
    relation_batch = torch.tensor([0, 1, 2])

    # Get the embeddings
    embeddings = conve_model.relation_embeddings(relation_batch)

    # Check if the output is of the correct type
    assert is_float_tensor_with_dim(embeddings, dim=2)

    # Check if the shape is correct
    expected_shape = (3, conve_model.config.embedding_dim)
    assert embeddings.shape == expected_shape

    # Check if the output is a FloatTensor
    assert isinstance(embeddings, torch.FloatTensor)


def test_save_load(conve_model: ConvE):
    tmp_path = Path(".tests_tmp")
    save_path = tmp_path / "conve_model"
    try:
        save_path.mkdir(parents=True)

        conve_model.save(save_path)

        loaded_model = ConvE.load(save_path)

        assert isinstance(loaded_model, ConvE)
        assert loaded_model.config.num_entities == conve_model.config.num_entities
        assert loaded_model.config.num_relations == conve_model.config.num_relations
        assert loaded_model.config.embedding_dim == conve_model.config.embedding_dim

        # Test if the loaded model produces the same output as the original model
        subjects = torch.randint(0, 1000, (32,))
        relations = torch.randint(0, 50, (32,))
        objects = torch.randint(0, 1000, (32,))
        sro_batch = torch.stack([subjects, relations, objects], dim=1)
        conve_model.eval()
        loaded_model.eval()
        original_output = conve_model.forward(sro_batch)
        loaded_output = loaded_model.forward(sro_batch)

        assert torch.allclose(original_output.scores, loaded_output.scores)
    finally:
        # Remove the temporary directory
        for f in save_path.iterdir():
            f.unlink()
        save_path.rmdir()
