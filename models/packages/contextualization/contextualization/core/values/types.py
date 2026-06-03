from enum import StrEnum


class ReferenceDatabaseType(StrEnum):
    NDJSON = "ndjson"
    POSTGRES = "postgres"


class ReferenceReaderType(StrEnum):
    PARQUET = "parquet"
    NDJSON = "ndjson"


class PublicationVectorDatabaseType(StrEnum):
    MILVUS = "milvus"
    IN_MEMORY = "in_memory"


class PublicationEncoderType(StrEnum):
    LLM = "llm"


class PublicationScorerType(StrEnum):
    COVERAGE = "coverage"
    RECENCY = "recency"
    AGGREGATED = "aggregated"


class RetrieverType(StrEnum):
    LOOKUP = "lookup"


class ContextSummarizerType(StrEnum):
    LLM = "llm"
