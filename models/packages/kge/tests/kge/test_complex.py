from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from hakken_ml_toolkit.ml_base_structures.data_generator import DummyDataGenerator

from kge.common.actions.kge_loader_action import KGELoader
from kge.common.constants import BaseFolderName
from kge.common.validation import is_float_tensor_with_dim
from kge.models.base import KGEI
from kge.models.complex import ComplEx, ComplExConfig


def complex_model(embedding_dim: int, num_entities: int, num_relations: int) -> ComplEx:
    config = ComplExConfig(
        embedding_dim=embedding_dim,
        num_entities=num_entities,
        num_relations=num_relations,
    )
    return ComplEx(config)


@pytest.mark.parametrize("embedding_dim", [5])
@pytest.mark.parametrize("num_entities", [10])
@pytest.mark.parametrize("num_relations", [5])
def test_complex_initialization(embedding_dim: int, num_entities: int, num_relations: int) -> None:
    model = complex_model(embedding_dim, num_entities, num_relations)
    assert isinstance(model, ComplEx)
    assert model.embedding_dim() == embedding_dim * 2
    assert model._entity_embeddings.num_embeddings == num_entities
    assert model._relation_embeddings.num_embeddings == num_relations
    assert model._entity_embeddings.embedding_dim == 2 * embedding_dim
    assert model._relation_embeddings.embedding_dim == 2 * embedding_dim


@pytest.mark.parametrize("batch_size", [3])
@pytest.mark.parametrize("embedding_dim", [5])
@pytest.mark.parametrize("num_entities", [10])
@pytest.mark.parametrize("num_relations", [5])
def test_complex_split_complex(
    batch_size: int, embedding_dim: int, num_entities: int, num_relations: int
) -> None:
    model = complex_model(embedding_dim, num_entities, num_relations)
    x = torch.randn(batch_size, 2 * embedding_dim)
    re, im = model._split_complex(x)
    assert re.shape == (batch_size, embedding_dim)
    assert im.shape == (batch_size, embedding_dim)
    assert torch.allclose(torch.cat([re, im], dim=-1), x)


@pytest.mark.parametrize("batch_size", [3])
@pytest.mark.parametrize("embedding_dim", [5])
@pytest.mark.parametrize("num_entities", [10])
@pytest.mark.parametrize("num_relations", [5])
def test_complex_score_subjects(
    batch_size: int, embedding_dim: int, num_entities: int, num_relations: int
) -> None:
    model = complex_model(embedding_dim, num_entities, num_relations)
    ro_batch = DummyDataGenerator.ro_batch(
        batch_size, num_entities=num_entities, num_relations=num_relations, device="cpu"
    )

    scores = model.score_subjects(ro_batch)

    assert is_float_tensor_with_dim(scores, dim=2)
    assert scores.shape == (batch_size, num_entities)


@pytest.mark.parametrize("batch_size", [3])
@pytest.mark.parametrize("embedding_dim", [5])
@pytest.mark.parametrize("num_entities", [10])
@pytest.mark.parametrize("num_relations", [5])
def test_complex_score_relations(
    batch_size: int, embedding_dim: int, num_entities: int, num_relations: int
) -> None:
    model = complex_model(embedding_dim, num_entities, num_relations)
    so_batch = DummyDataGenerator.so_batch(batch_size, num_entities=num_entities, device="cpu")

    scores = model.score_relations(so_batch)

    assert is_float_tensor_with_dim(scores, dim=2)
    assert scores.shape == (batch_size, num_relations)


@pytest.mark.parametrize("batch_size", [3])
@pytest.mark.parametrize("embedding_dim", [5])
@pytest.mark.parametrize("num_entities", [10])
@pytest.mark.parametrize("num_relations", [5])
def test_complex_score_objects(
    batch_size: int, embedding_dim: int, num_entities: int, num_relations: int
) -> None:
    model = complex_model(embedding_dim, num_entities, num_relations)
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
def test_complex_score(
    batch_size: int, embedding_dim: int, num_entities: int, num_relations: int
) -> None:
    model = complex_model(embedding_dim, num_entities, num_relations)
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
def test_complex_forward(
    batch_size: int, embedding_dim: int, num_entities: int, num_relations: int
) -> None:
    model = complex_model(embedding_dim, num_entities, num_relations)
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
def test_complex_entity_embeddings(
    batch_size: int, embedding_dim: int, num_entities: int, num_relations: int
) -> None:
    model = complex_model(embedding_dim, num_entities, num_relations)
    entity_batch = DummyDataGenerator.entity_batch(
        batch_size=batch_size, num_entities=num_entities, device="cpu"
    )

    embeddings = model.entity_embeddings(entity_batch)

    assert is_float_tensor_with_dim(embeddings, dim=2)
    assert embeddings.shape == (batch_size, 2 * embedding_dim)


@pytest.mark.parametrize("batch_size", [3])
@pytest.mark.parametrize("embedding_dim", [5])
@pytest.mark.parametrize("num_entities", [10])
@pytest.mark.parametrize("num_relations", [5])
def test_complex_relation_embeddings(
    batch_size: int, embedding_dim: int, num_entities: int, num_relations: int
) -> None:
    model = complex_model(embedding_dim, num_entities, num_relations)
    relation_batch = DummyDataGenerator.relation_batch(
        batch_size=batch_size, num_relations=num_relations, device="cpu"
    )

    embeddings = model.relation_embeddings(relation_batch)

    assert is_float_tensor_with_dim(embeddings, dim=2)
    assert embeddings.shape == (batch_size, 2 * embedding_dim)


@pytest.mark.parametrize("batch_size", [3])
@pytest.mark.parametrize("embedding_dim", [5])
@pytest.mark.parametrize("num_entities", [10])
@pytest.mark.parametrize("num_relations", [5])
def test_complex_save_and_load(
    batch_size: int,
    embedding_dim: int,
    num_entities: int,
    num_relations: int,
    tmp_path: Path,
) -> None:
    model = complex_model(embedding_dim, num_entities, num_relations)

    # Save the model
    save_path = tmp_path / "complex_model"
    save_path.mkdir()
    model.save(save_path)

    # Load the model
    loaded_model = ComplEx.load(save_path)

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


@pytest.fixture
def mock_yaml_utils():
    with patch("kge.common.actions.kge_loader_action.YAMLUtils") as mock:
        yield mock


@pytest.fixture
def mock_hydra():
    with patch("kge.common.actions.kge_loader_action.hydra") as mock:
        yield mock


def test_kge_simple_loader(tmp_path: Path, mock_yaml_utils, mock_hydra):
    root_path = tmp_path / "model"
    root_path.mkdir()
    (root_path / BaseFolderName.KGE).mkdir()
    (root_path / ".hydra").mkdir()
    config_path = root_path / ".hydra" / "config.yaml"

    config_path.write_text("model:\n  _target_: dummy.path.to.KGEModel")

    mock_config = {"model": {"_target_": "kge.models.complex.ComplEx"}}
    mock_yaml_utils.load.return_value = mock_config

    mock_kge_class = MagicMock()
    mock_kge_instance = MagicMock(spec=KGEI)
    mock_kge_class.load.return_value = mock_kge_instance
    mock_hydra.utils.get_class.return_value = mock_kge_class

    loaded_model = KGELoader.load(str(root_path))

    assert loaded_model == mock_kge_instance
    mock_yaml_utils.load.assert_called_once_with(config_path)
    mock_hydra.utils.get_class.assert_called_once_with("kge.models.complex.ComplEx")
    mock_kge_class.load.assert_called_once_with(root_path / BaseFolderName.KGE, device="cpu")
