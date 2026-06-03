"""Euclidean similarity: 1/(1 + L2 distance). Range (0, 1], higher = more similar."""

import numpy as np


def compute_similarity_matrix(
    embeddings_1: list[list[float]],
    embeddings_2: list[list[float]] | None = None,
) -> list[list[float]]:
    """Pairwise euclidean-based similarity between embedding vectors.

    Similarity = 1/(1 + distance); identical vectors yield 1.0, bounded (0, 1].

    If embeddings_2 is None, returns (n1, n1) similarity within embeddings_1.
    If both are provided, returns (n1, n2) similarity between embeddings_1 and
    embeddings_2.

    Args:
        embeddings_1: First list of embedding vectors.
        embeddings_2: Optional second list. If None, similarity within embeddings_1.

    Returns:
        Similarity matrix; entry [i][j] is similarity of the i-th vector in
        the first set and the j-th in the second (or within embeddings_1).
    """
    arr1 = np.array(embeddings_1, dtype=np.float64)
    sqnorms1 = np.sum(arr1 * arr1, axis=1)

    if embeddings_2 is None:
        dot = np.dot(arr1, arr1.T)
        sq_dists = np.maximum(sqnorms1[:, None] + sqnorms1[None, :] - 2 * dot, 0.0)
    else:
        arr2 = np.array(embeddings_2, dtype=np.float64)
        sqnorms2 = np.sum(arr2 * arr2, axis=1)
        dot = np.dot(arr1, arr2.T)
        sq_dists = np.maximum(sqnorms1[:, None] + sqnorms2[None, :] - 2 * dot, 0.0)

    distances = np.sqrt(sq_dists)
    similarity = 1.0 / (1.0 + distances)
    return similarity.tolist()
