from strenum import StrEnum


class ScoreType(StrEnum):
    NECESSARY = "necessary"
    SUFFICIENT = "sufficient"


class CandidateFinderType(StrEnum):
    CORPUS = "corpus"
    LATENT = "latent"


class RerankStrategy(StrEnum):
    UNIQUE_PATHWAYS = "unique_pathways"
    SCORES = "scores"
