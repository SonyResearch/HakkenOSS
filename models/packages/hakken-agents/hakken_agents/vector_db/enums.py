from strenum import StrEnum


class SimilarityMetric(StrEnum):
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
