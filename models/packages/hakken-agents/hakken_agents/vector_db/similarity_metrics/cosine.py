"""Cosine similarity: L2-normalized dot product. Range [-1, 1], higher = more similar."""

import numpy as np


def compute_similarity_matrix(
    embeddings_1: list[list[float]],
    embeddings_2: list[list[float]] | None = None,
) -> list[list[float]]:
    """Pairwise cosine similarity between embedding vectors.

    If embeddings_2 is None, returns (n1, n1) similarity within embeddings_1.
    If both are provided, returns (n1, n2) similarity between embeddings_1 and
    embeddings_2.

    Args:
        embeddings_1: First list of embedding vectors.
        embeddings_2: Optional second list. If None, similarity within embeddings_1.

    Returns:
        Similarity matrix; entry [i][j] is cosine similarity of the i-th vector
        in the first set and the j-th in the second (or within embeddings_1).
    """
    arr1 = np.array(embeddings_1, dtype=np.float64)
    norms1 = np.linalg.norm(arr1, axis=1, keepdims=True)
    norms1 = np.where(norms1 == 0, 1.0, norms1)
    normalized_1 = arr1 / norms1

    if embeddings_2 is None:
        matrix = np.dot(normalized_1, normalized_1.T)
    else:
        arr2 = np.array(embeddings_2, dtype=np.float64)
        norms2 = np.linalg.norm(arr2, axis=1, keepdims=True)
        norms2 = np.where(norms2 == 0, 1.0, norms2)
        normalized_2 = arr2 / norms2
        matrix = np.dot(normalized_1, normalized_2.T)

    return matrix.tolist()
