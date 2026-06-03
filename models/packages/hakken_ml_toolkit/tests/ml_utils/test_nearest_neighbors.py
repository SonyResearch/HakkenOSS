import pytest

try:
    import torch
except ImportError:
    pytest.skip("PyTorch is not installed", allow_module_level=True)


from hakken_ml_toolkit.ml_utils.extras.nearest_neighbors import NearestNeighbors


@pytest.fixture
def sample_embeddings():
    return torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.5, 0.5, 0.5]])


def test_embeddings_l2(sample_embeddings: torch.Tensor):
    query = torch.tensor([[1.0, 0.0, 0.0]])

    distances, indices = NearestNeighbors.embeddings_l2(query, sample_embeddings)

    assert isinstance(distances, torch.Tensor)
    assert isinstance(indices, torch.Tensor)
    assert distances.shape == (1,)
    assert indices.shape == (1,)
    assert indices[0] == 0
    assert torch.isclose(distances[0], torch.tensor(0.0))


def test_embeddings_l2_multiple_queries(sample_embeddings: torch.Tensor):
    queries = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.3, 0.3, 0.3]])

    distances, indices = NearestNeighbors.embeddings_l2(queries, sample_embeddings)

    assert distances.shape == (3,)
    assert indices.shape == (3,)
    assert torch.all(indices == torch.tensor([0, 1, 3]))


def test_embeddings_l2_top_k_not_implemented(sample_embeddings: torch.Tensor):
    query = torch.tensor([[1.0, 0.0, 0.0]])

    with pytest.raises(NotImplementedError):
        NearestNeighbors.embeddings_l2(query, sample_embeddings, top_k=2)


def test_embeddings_cosine(sample_embeddings: torch.Tensor):
    query = torch.tensor([[1.0, 1.0, 0.0]])

    distances, indices = NearestNeighbors.embeddings_cosine(query, sample_embeddings)

    assert isinstance(distances, torch.Tensor)
    assert isinstance(indices, torch.Tensor)
    assert distances.shape == (1,)
    assert indices.shape == (1,)
    assert indices[0] == 3  # The [0.5, 0.5, 0.5] vector should be most similar


def test_embeddings_cosine_0_distance(sample_embeddings: torch.Tensor):
    query = torch.tensor([[0.5, 0.5, 0.5]])

    similarity, indices = NearestNeighbors.embeddings_cosine(query, sample_embeddings)

    assert isinstance(similarity, torch.Tensor)
    assert isinstance(indices, torch.Tensor)
    assert similarity.shape == (1,)
    assert indices.shape == (1,)
    assert indices[0] == 3  # The [0.5, 0.5, 0.5] vector should be most similar
    assert torch.isclose(similarity[0], torch.tensor(1.0), atol=1e-4)


def test_embeddings_cosine_multiple_queries(sample_embeddings: torch.Tensor):
    queries = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.5]])

    distances, indices = NearestNeighbors.embeddings_cosine(queries, sample_embeddings)

    assert distances.shape == (3,)
    assert indices.shape == (3,)
    assert torch.all(indices == torch.tensor([0, 1, 3]))


def test_embeddings_cosine_top_k_not_implemented(sample_embeddings: torch.Tensor):
    query = torch.tensor([[1.0, 0.0, 0.0]])

    with pytest.raises(NotImplementedError):
        NearestNeighbors.embeddings_cosine(query, sample_embeddings, top_k=2)


def test_embeddings_l2_and_cosine_consistency(sample_embeddings: torch.Tensor):
    query = torch.tensor([[0.1, 0.2, 0.3]])

    _, l2_indices = NearestNeighbors.embeddings_l2(query, sample_embeddings)
    _, cosine_indices = NearestNeighbors.embeddings_cosine(query, sample_embeddings)

    assert l2_indices[0] == cosine_indices[0]
