from pathlib import Path

import pytest
import torch
from hakken_ml_toolkit.ml_base_structures.data_generator import DummyDataGenerator

from kge.common.validation import is_float_tensor_with_dim
from kge.models.distmult import DistMult, DistMultConfig


def distmult_model(embedding_dim: int, num_entities: int, num_relations: int) -> DistMult:
    config = DistMultConfig(
        embedding_dim=embedding_dim,
        num_entities=num_entities,
        num_relations=num_relations,
    )
    return DistMult(config)


@pytest.mark.parametrize("embedding_dim", [5])
@pytest.mark.parametrize("num_entities", [10])
@pytest.mark.parametrize("num_relations", [5])
def test_distmult_initialization(embedding_dim: int, num_entities: int, num_relations: int) -> None:
    model = distmult_model(embedding_dim, num_entities, num_relations)
    assert isinstance(model, DistMult)
    assert model.embedding_dim() == embedding_dim
    assert model._entity_embeddings.num_embeddings == num_entities
    assert model._relation_embeddings.num_embeddings == num_relations


@pytest.mark.parametrize("batch_size", [3])
@pytest.mark.parametrize("embedding_dim", [5])
@pytest.mark.parametrize("num_entities", [10])
@pytest.mark.parametrize("num_relations", [5])
def test_distmult_score_objects(
    batch_size: int, embedding_dim: int, num_entities: int, num_relations: int
) -> None:
    model = distmult_model(embedding_dim, num_entities, num_relations)
    sr_batch = DummyDataGenerator.sr_batch(
        batch_size, num_entities=num_entities, num_relations=num_relations, device="cpu"
    )

    scores = model.score_objects(sr_batch)

    assert is_float_tensor_with_dim(scores, dim=2)
    assert scores.shape == (batch_size, num_entities)


@pytest.mark.parametrize("batch_size", [3])
@pytest.mark.parametrize("embedding_dim", [5])
@pytest.mark.parametrize("num_entities", [10])
@pytest.mark.parametrize("num_relations", [5])
def test_distmult_score(
    batch_size: int, embedding_dim: int, num_entities: int, num_relations: int
) -> None:
    model = distmult_model(embedding_dim, num_entities, num_relations)
    sro_batch = DummyDataGenerator.sro_batch(
        batch_size=batch_size,
        num_entities=num_entities,
        num_relations=num_relations,
        device="cpu",
    )

    scores = model.score(sro_batch)

    assert is_float_tensor_with_dim(scores, dim=2)
    assert scores.shape == (batch_size, 1)


@pytest.mark.parametrize("batch_size", [3])
@pytest.mark.parametrize("embedding_dim", [5])
@pytest.mark.parametrize("num_entities", [10])
@pytest.mark.parametrize("num_relations", [5])
def test_distmult_forward(
    batch_size: int, embedding_dim: int, num_entities: int, num_relations: int
) -> None:
    model = distmult_model(embedding_dim, num_entities, num_relations)
    sro_batch = DummyDataGenerator.sro_batch(
        batch_size=batch_size,
        num_entities=num_entities,
        num_relations=num_relations,
        device="cpu",
    )

    output = model.forward(sro_batch)

    assert output.scores.shape == (batch_size, 1)


@pytest.mark.parametrize("batch_size", [3])
@pytest.mark.parametrize("embedding_dim", [5])
@pytest.mark.parametrize("num_entities", [10])
@pytest.mark.parametrize("num_relations", [5])
def test_distmult_entity_embeddings(
    batch_size: int, embedding_dim: int, num_entities: int, num_relations: int
) -> None:
    model = distmult_model(embedding_dim, num_entities, num_relations)
    entity_batch = DummyDataGenerator.entity_batch(
        batch_size=batch_size, num_entities=num_entities, device="cpu"
    )

    embeddings = model.entity_embeddings(entity_batch)

    assert is_float_tensor_with_dim(embeddings, dim=2)
    assert embeddings.shape == (batch_size, embedding_dim)


@pytest.mark.parametrize("batch_size", [3])
@pytest.mark.parametrize("embedding_dim", [5])
@pytest.mark.parametrize("num_entities", [10])
@pytest.mark.parametrize("num_relations", [5])
def test_distmult_relation_embeddings(
    batch_size: int, embedding_dim: int, num_entities: int, num_relations: int
) -> None:
    model = distmult_model(embedding_dim, num_entities, num_relations)
    relation_batch = DummyDataGenerator.relation_batch(
        batch_size=batch_size, num_relations=num_relations, device="cpu"
    )

    embeddings = model.relation_embeddings(relation_batch)

    assert is_float_tensor_with_dim(embeddings, dim=2)
    assert embeddings.shape == (batch_size, embedding_dim)


@pytest.mark.parametrize("batch_size", [3])
@pytest.mark.parametrize("embedding_dim", [5])
@pytest.mark.parametrize("num_entities", [10])
@pytest.mark.parametrize("num_relations", [5])
def test_distmult_save_and_load(
    batch_size: int,
    embedding_dim: int,
    num_entities: int,
    num_relations: int,
    tmp_path: Path,
) -> None:
    model = distmult_model(embedding_dim, num_entities, num_relations)

    # Save the model
    save_path = tmp_path / "distmult_model"
    save_path.mkdir()
    model.save(save_path)

    # Load the model
    loaded_model: DistMult = DistMult.load(save_path)

    # Check if the loaded model has the same configuration
    assert loaded_model.config.embedding_dim == model.config.embedding_dim
    assert loaded_model.config.num_entities == model.config.num_entities
    assert loaded_model.config.num_relations == model.config.num_relations

    # Check if the loaded model produces the same outputs
    sro_batch = DummyDataGenerator.sro_batch(
        batch_size=batch_size,
        num_entities=num_entities,
        num_relations=num_relations,
        device="cpu",
    )

    original_output = model.forward(sro_batch)
    loaded_output = loaded_model.forward(sro_batch)

    assert torch.allclose(original_output.scores, loaded_output.scores)
