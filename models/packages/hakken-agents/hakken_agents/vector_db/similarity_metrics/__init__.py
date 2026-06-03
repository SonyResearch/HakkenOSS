"""Pairwise similarity metrics for embedding vectors."""

from hakken_agents.vector_db.enums import SimilarityMetric
from hakken_agents.vector_db.similarity_metrics.cosine import (
    compute_similarity_matrix as cosine_similarity_matrix,
)
from hakken_agents.vector_db.similarity_metrics.dot_product import (
    compute_similarity_matrix as dot_product_similarity_matrix,
)
from hakken_agents.vector_db.similarity_metrics.euclidean import (
    compute_similarity_matrix as euclidean_similarity_matrix,
)


def compute_similarity_matrix(
    embeddings_1: list[list[float]],
    embeddings_2: list[list[float]] | None = None,
    metric: SimilarityMetric = SimilarityMetric.COSINE,
) -> list[list[float]]:
    """Compute pairwise similarity matrix for the given embeddings and metric.

    If embeddings_2 is None, returns (n1, n1) similarity within embeddings_1.
    If both are provided, returns (n1, n2) similarity between embeddings_1 and
    embeddings_2.

    Args:
        embeddings_1: First list of embedding vectors.
        embeddings_2: Optional second list. If None, similarity within embeddings_1.
        metric: One of COSINE, EUCLIDEAN, DOT_PRODUCT.

    Returns:
        Similarity matrix; entry [i][j] is the similarity between the i-th
        vector in the first set and the j-th in the second (or within set 1).
    """
    if metric == SimilarityMetric.COSINE:
        return cosine_similarity_matrix(embeddings_1, embeddings_2)
    if metric == SimilarityMetric.EUCLIDEAN:
        return euclidean_similarity_matrix(embeddings_1, embeddings_2)
    if metric == SimilarityMetric.DOT_PRODUCT:
        return dot_product_similarity_matrix(embeddings_1, embeddings_2)
