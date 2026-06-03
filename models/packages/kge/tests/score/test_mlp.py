import pytest
import torch
from gnn import MLP, MLPConfig

from kge.common.types import FloatTensor2D
from kge.scores import MLPScore


@pytest.fixture
def embedding_dim() -> int:
    return 4


@pytest.fixture
def batch_size() -> int:
    return 2


@pytest.fixture
def num_entities() -> int:
    return 3


@pytest.fixture
def num_relations() -> int:
    return 5


@pytest.fixture
def mlp_score(embedding_dim: int) -> MLPScore:
    return MLPScore(MLPConfig(input_dim=3 * embedding_dim, num_layers=1, output_dim=1))


@pytest.fixture
def mock_embeddings(
    embedding_dim: int, batch_size: int, num_entities: int, num_relations: int
) -> tuple[FloatTensor2D, FloatTensor2D, FloatTensor2D, FloatTensor2D, FloatTensor2D]:
    s_emb = torch.randn(batch_size, embedding_dim)
    r_emb = torch.randn(batch_size, embedding_dim)
    o_emb = torch.randn(batch_size, embedding_dim)
    entity_embeddings = torch.randn(num_entities, embedding_dim)
    relation_embeddings = torch.randn(num_relations, embedding_dim)

    return s_emb, r_emb, o_emb, entity_embeddings, relation_embeddings


def test_init(embedding_dim: int) -> None:
    score = MLPScore(MLPConfig(input_dim=3 * embedding_dim, num_layers=1, output_dim=1))
    assert isinstance(score.mlp, MLP)
    assert score.mlp.config.input_dim == 3 * embedding_dim
    assert score.mlp.config.output_dim == 1


def test_subjects(
    mlp_score: MLPScore,
    mock_embeddings: tuple[
        FloatTensor2D, FloatTensor2D, FloatTensor2D, FloatTensor2D, FloatTensor2D
    ],
    batch_size: int,
    num_entities: int,
) -> None:
    _, r_emb, o_emb, entity_embeddings, _ = mock_embeddings
    scores = mlp_score.subjects(r_emb, o_emb, entity_embeddings)

    assert isinstance(scores, torch.Tensor)
    assert scores.shape == (batch_size, num_entities)
    assert not torch.isnan(scores).any()


def test_relations(
    mlp_score: MLPScore,
    mock_embeddings: tuple[
        FloatTensor2D, FloatTensor2D, FloatTensor2D, FloatTensor2D, FloatTensor2D
    ],
    batch_size: int,
    num_relations: int,
) -> None:
    s_emb, _, o_emb, _, relation_embeddings = mock_embeddings
    scores = mlp_score.relations(s_emb, o_emb, relation_embeddings)

    assert isinstance(scores, torch.Tensor)
    assert scores.shape == (batch_size, num_relations)
    assert not torch.isnan(scores).any()


def test_objects(
    mlp_score: MLPScore,
    mock_embeddings: tuple[
        FloatTensor2D, FloatTensor2D, FloatTensor2D, FloatTensor2D, FloatTensor2D
    ],
    batch_size: int,
    num_entities: int,
) -> None:
    s_emb, r_emb, _, entity_embeddings, _ = mock_embeddings
    scores = mlp_score.objects(s_emb, r_emb, entity_embeddings)

    assert isinstance(scores, torch.Tensor)
    assert scores.shape == (batch_size, num_entities)
    assert not torch.isnan(scores).any()


def test_all(
    mlp_score: MLPScore,
    mock_embeddings: tuple[
        FloatTensor2D, FloatTensor2D, FloatTensor2D, FloatTensor2D, FloatTensor2D
    ],
    batch_size: int,
) -> None:
    s_emb, r_emb, o_emb, _, _ = mock_embeddings
    scores = mlp_score.all(s_emb, r_emb, o_emb)

    assert isinstance(scores, torch.Tensor)
    assert scores.shape == (batch_size, 1)
    assert not torch.isnan(scores).any()


def test_empty_batch(mlp_score: MLPScore, embedding_dim: int, num_entities: int) -> None:
    empty_batch = 0
    r_emb = torch.randn(empty_batch, embedding_dim)
    o_emb = torch.randn(empty_batch, embedding_dim)
    entity_embeddings = torch.randn(num_entities, embedding_dim)

    scores = mlp_score.subjects(r_emb, o_emb, entity_embeddings)
    assert scores.shape == (empty_batch, num_entities)
