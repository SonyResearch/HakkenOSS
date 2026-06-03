"""Dot-product similarity: raw inner product. Unbounded; higher = more similar."""

import numpy as np


def compute_similarity_matrix(
    embeddings_1: list[list[float]],
    embeddings_2: list[list[float]] | None = None,
) -> list[list[float]]:
    """Pairwise dot product between embedding vectors.

    If embeddings_2 is None, returns (n1, n1) similarity within embeddings_1.
    If both are provided, returns (n1, n2) similarity between embeddings_1 and
    embeddings_2.

    Args:
        embeddings_1: First list of embedding vectors.
        embeddings_2: Optional second list. If None, similarity within embeddings_1.

    Returns:
        Similarity matrix; entry [i][j] is dot product of the i-th vector in
        the first set and the j-th in the second (or within embeddings_1).
    """
    arr1 = np.array(embeddings_1, dtype=np.float64)
    if embeddings_2 is None:
        matrix = np.dot(arr1, arr1.T)
    else:
        arr2 = np.array(embeddings_2, dtype=np.float64)
        matrix = np.dot(arr1, arr2.T)
    return matrix.tolist()
